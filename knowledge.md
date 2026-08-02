# Knowledge Base
Last updated: 2026-08-02T20:00:07.684819

## [dis-20260620063206-7fc4ae]
Category: discovery
Tags: threat-model, security, pylasdev, stride
Changed: 2026-06-20T06:32:06.034875

pylasdev-reborn v1.6.0 threat model: 1 HIGH (regex ReDoS in DATA_LINE_PATTERN parser.py:54), 3 MEDIUM (allocation bounds too high, missing line-length guard, parse ambiguity), 5 LOW. No CRITICAL. No code exec primitives (confirmed: zero eval/exec/pickle). Trust boundary at file→library ingestion (encoding.py:73). Good posture: size limits, NaN guards, encoding fallback, writer sanitization. Gaps: regex backtracking, no per-line length limit, MAX_TOTAL_ELEMENTS=1B too high for typical hosts.

## [dis-20260628204235-721d54]
Category: discovery
Tags: pylasdev-reborn, production-check, sanitization
Changed: 2026-06-28T20:42:35.390604

pylasdev-reborn production check complete (2026-06-28). 11 confirmed findings fixed across 2 fix passes. Writer sanitization: _sanitize_las_value() now applied to ALL LAS structural metadata fields (vers, dlm, well keys/values, curve mnemonic/unit/api_code/desc, param mnemonic/unit/value/desc, section.name, curve names in ~A header). Test hardening: T1-T7 strengthened weak/missing assertions, T4 fixed flaky concurrent test (pytest.fail -> warnings.warn). Final: 296/296 tests, 91.22% cov, ruff+mypy clean. Commit 25a83a6 pushed.

## [con-20260629034557-d3f225]
Category: context
Tags: pylasdev-reborn, production-check, completed
Changed: 2026-06-29T03:45:57.460270

COMPLETED: pylasdev-reborn production check. 19 agents across 7 stages. 40 findings (15 MEDIUM, 25 LOW) → 3 confirmed coverage gaps fixed (4 new tests). 300/300 tests pass. Committed 1fd6491 and pushed. No source code changes needed — all findings were test coverage gaps or intentional design.

## [dis-20260630183719-23704c]
Category: discovery
Tags: pylasdev, production-check, bugs, las30
Changed: 2026-06-30T18:37:19.554096

pylasdev-reborn production check (2026-06-30): Found and fixed 9 bugs. Most critical: F-01 (LAS 3.0 structured data sections like ~Core_Data, ~Drilling_Data silently dropped due to SECTION_PATTERN capturing only first letter of section name). Fix: changed regex to ^~(\S+)(.*) and added full section word classification. Also fixed: LAS 1.2 field swap, WRAP mode preservation, section_type metadata for LAS 3.0 data sections, short-row warning asymmetry, pathological wrap misalignment detection.

## [dis-20260630201323-9a144e]
Category: discovery
Tags: pylasdev, production-check, findings
Changed: 2026-06-30T20:13:23.644353

pylasdev-reborn production check complete: 11 findings fixed (2 HIGH, 9 MEDIUM). LAS 1.2 value/description swap was unconditional (F-01), _DATA_SECTION_WORDS missed LAS 3.0 section types (F-03), unknown sections left stale handler state (F-02/F-08), ~Other was emitted for LAS 3.0 (F-05), wrap detection had false positive with comma delimiter (F-06). Tests grew from 300 to 311.

 emission in writer, and _section_curve_end_idx capping in parser.

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

## [pat-20260701003853-ea0153]
Category: pattern
Tags: pylasdev, production-check
Changed: 2026-07-01T00:38:53.395049

pylasdev-reborn production check: 44 agents, 28 findings fixed across 2 FIX convergence passes. Key areas: parser mnemonic collision (first-wins), delimiter-aware data reading (maxsplit guards), writer precision formatting (_format_fixed_precision for e-format), DEV delimiter auto-detection (>= for tie case). 364 tests pass.

## [con-20260701005900-8cc26b]
Category: context
Tags: dev-format, research, pylasdev, specification
Changed: 2026-07-01T00:59:00.535108

DEV format research report at tmp/dev-format-research-report.md — covers all 6 questions with sources and confidence tiers: (1) no single standard, LAS 3.0 Inclinometry + Petrel .dev are closest to authoritative, (2) MD/INC/AZI and MD/TVD/X/Y paradigms, (3) multiple delimiter conventions, (4) null sentinel = -999.25 in LAS, -999 common, (5) 5 calculation methods but NONE encoded in file format, (6) structural comparison table with parsing implications for pylasdev

## [dis-20260701013949-d433be]
Category: discovery
Tags: pylasdev, production-check, las, dev, spec-compliance
Changed: 2026-07-01T01:39:49.031766

pylasdev-reborn production check complete: 385 tests pass, ruff/mypy clean. Fixed DEV reader format auto-detection (DUG Insight + headerless), delimiter index bug, and special float guard. 22 agents, 2 fix iterations. Research reports at tmp/s1-research-las-report.md and tmp/dev-format-research-report.md provide authoritative LAS/DEV spec references.

## [dis-20260714200336-c471c3]
Category: discovery
Tags: pylasdev, production-check, completed
Changed: 2026-07-14T20:03:36.516447

pylasdev-reborn production check completed: 17 agents, 9 stages. Fixed 3 confirmed findings (F-01 HIGH: WRAP=YES header override, F-02 MEDIUM: LAS 1.2 well field heuristic + descriptions dict, F-04 MEDIUM: depth_had_extra stale flag reset). 398/398 tests pass, 91.99% coverage. Committed b22ba33, pushed to origin/master.

## [dis-20260714210626-91db69]
Category: discovery
Tags: pylasdev, production-check, las3
Changed: 2026-07-14T21:06:26.562787

pylasdev-reborn production check completed: LAS 3.0 custom section type roundtrip fix (parser preserves _DATA-suffixed section words as section_type, writer uses type name as its own prefix). 16 findings discovered (1 HIGH→MEDIUM, 5 MEDIUM→4 LOW, 10 LOW). All 398 tests pass. Commit a604ad6 pushed to origin/master.

## [dis-20260714232105-5f2863]
Category: discovery
Tags: pylasdev, production-check, summary
Changed: 2026-07-14T23:21:05.712568

pylasdev-reborn production check (2026-07-14): 406→407 tests, 52 findings (1 HIGH, 15 MEDIUM, 36 LOW), 5 confirmed+fixed. Key fixes: DLM for LAS 2.0, dev_reader splitlines sanitization, RuntimeError for internal invariant, LAS 1.2 description preservation, LAS 3.0 ~Other deprecation test. 92% coverage, ruff+mypy clean.

## [dis-20260715015107-0d9932]
Category: discovery
Tags: pylasdev, production-check, audit
Changed: 2026-07-15T01:51:07.941447

pylasdev-reborn production check: 101 agents, 67 fixes (2 HIGH, 59 MEDIUM), 26 new tests, 433/433 pass, 92.11% cov. Key fixes: well descriptions emission, Fortran D-notation, encoding quality-based selection with UTF-8 tiebreaker, DEFAULT_MAX_FILE_SIZE, NaN-aware comparison, source_file persistence, well_format API exposure, Definition prefix sanitization

## [dis-20260715090802-5aab5e]
Category: discovery
Tags: pylasdev, production-check, parser, las
Changed: 2026-07-15T09:08:02.743851

pylasdev-reborn production check completed: 47-agent workflow found 68 issues (3 HIGH, 29 MEDIUM, 36 LOW). After adversarial verification: 16 CONFIRMED MEDIUM+ → 17 fixes applied (1 HIGH data corruption bug in parser.py H-03, 15 MEDIUM fixes + 1 convergence fix). All 433 tests pass, ruff/mypy clean. Key finding: parser.py had a definition-range save ordering bug where pipe-target lookup ran BEFORE range save, causing silent data corruption in LAS 3.0 files with ~C→_Definition sequences.

## [pat-20260717091255-e2d324]
Category: pattern
Tags: python, security, output, sanitization, writer
Changed: 2026-07-17T09:12:55.159015

Output sanitization completeness: When N out of M output fields/call-sites apply sanitization, the single unsanitized field is the injection vector. In pylasdev, 19/19 call sites used `_sanitize_las_value()` — except one (`actual_wrap` in writer.py:168). A crafted WRAP value with `\n~` injects fake section headers. Checklist: enumerate ALL output fields rendered to text; verify every one passes through the sanitizer. One gap negates all others.

## [pat-20260717091257-1895b1]
Category: pattern
Tags: python, review, consistency, writer, models
Changed: 2026-07-17T09:12:57.889725

Inconsistent sibling code-path guards: When a guard (check, validation, raise) exists in one code path but not in a structurally similar sibling path handling the same class of input, the gap is almost certainly a bug. In pylasdev: `_write_version_section` computed `is_las30` but not `is_las12` while sibling `_write_well_section` computed both (F-014). And `data_sections` silently `continue`d on non-dict elements while sibling paths `parameter_details` and `params` raised `TypeError` (F-058). Checklist: for every guard present in one sibling path, grep for its equivalent in all sibling paths; an inconsistency is a finding.

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

## [got-20260717091307-7ea6ca]
Category: gotcha
Tags: python, encoding
Changed: 2026-07-17T09:13:07.117988

Encoding fallback tie-break must prefer domain-likely encodings. In pylasdev `_decode_best_quality`: without chardet, cp1251 (Cyrillic) and cp1252 (Western European) had identical word-char ratio scores; stable-sort by insertion order picked cp1251 — producing mojibake for Western European files (F-061). In geoscience/energy, cp1252/latin-1 is overwhelmingly more common than cp1251. Fix: when multiple encodings score equally in quality detection, break ties by domain prevalence (cp1252 > cp1251 for Western data, shift_jis > euc_jp for Japanese, etc.), not by stable sort order. Always prefer chardet when available.

## [pat-20260717091310-73a82e]
Category: pattern
Tags: python, fix, review, parser, cwls
Changed: 2026-07-17T09:13:10.170581

Incomplete fix across detection branches: When a bug exists in a feature with both an explicit (user-specified) code path and an auto-detection path, fixing only the explicit branch leaves the auto-detected branches broken. In pylasdev: the CWLS value/description inversion was fixed in the explicit`_well_format == 'cwls'` branch (F-004 fix), but two auto-detected CWLS branches (lines 957-958 and 967-968) had the identical inversion and were missed (R2-001). Checklist: after fixing a bug, grep for ALL code paths that handle the same format/condition/mode — including auto-detection branches, fallback paths, and default else-branches — and verify the fix applies to all of them.

## [pat-20260717164610-581fab]
Category: pattern
Tags: python, input-validation, dict, deserialization, truthiness
Changed: 2026-07-17T16:46:10.292697

Truthiness dispatch via `data.get(key) or default`: The pattern `value = data.get('key') or default` silently accepts truthy wrong-type values (e.g., string 'abc', number 42) when the key exists but holds a non-None, non-empty falsy value or a truthy value of the wrong type. The `or` operator conflates 'key missing/None' with 'key present but wrong type' — a truthy non-dict `parameters` value produces zero parameters, truthy non-list `curves_data` silently falls into a legacy branch with data loss. Fix: extract a type-validating helper like `_resolve_dict_entry(data, key, expected_type, default_factory)` that does `in key` check, None check, and `isinstance` validation before returning, raising TypeError for wrong-type values. In pylasdev-reborn: `LASFile.from_dict()` had 5 `or`-pattern call sites producing a function-level regression hotspot (5 PRIOR_FIX_ATTEMPT findings across 4 prior commits that repeatedly patched leaf crashes without fixing root cause). The extracted `_resolve_dict_entry()` helper replaced all 5 sites, added mandatory `else` clause to params dispatch, and made 'key missing vs key present with wrong type' explicit.

## [got-20260717164613-322d5f]
Category: gotcha
Tags: python, parser, validation, state-machine
Changed: 2026-07-17T16:46:13.747235

Premature state flag before validation: Setting boolean success flags (`self._version_found = True`) at function entry before any line validation causes silent suppression of error conditions. When a flag is set unconditionally before `_match_data_line` returns, sections containing only garbage/non-matching lines are still marked 'found' — suppressing the 'missing mandatory section' error and producing spurious downstream warnings against incorrect defaults. Fix: move the flag set INSIDE the successful match branch, so it only fires when actual valid data was parsed. In pylasdev-reborn: `_parse_version()` at parser.py:861 set `_version_found = True` before any validation; if all lines in the ~V section failed to match, the flag stayed True, suppressing the real error.

## [pat-20260717164621-1e2924]
Category: pattern
Tags: python, format-detection, dev-reader, heuristic, parity
Changed: 2026-07-17T16:46:21.556396

Heuristic variant parity: When a format detection system has parallel heuristic variants (e.g., Pattern A and Pattern B for different format characteristics), ALL variants must have the same robustness fallbacks. In pylasdev-reborn `dev_reader.py`: DUG Pattern A had three fallback layers (count-match heuristic, count-mismatch fallback) added in prior fix rounds — Pattern B had NONE, silently failing on all-numeric column names. Similarly, simple-format detection using `all(_is_float_token(t))` on the first line incorrectly classified numeric-header files as headerless with zero fallback layers. Checklist: after adding robustness fallbacks to one heuristic variant, mechanically verify all sibling variants have equivalent protections. The gap between variants IS the bug.

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

## [got-20260717164652-acc693]
Category: gotcha
Tags: python, exception-handling, encoding, LookupError
Changed: 2026-07-17T16:46:52.149831

Exception hierarchy completeness: When wrapping library exceptions in domain-specific exceptions, catch the FULL parent class hierarchy, not just the specific subclass you expect. `except UnicodeDecodeError` misses `LookupError` (parent of `codecs.LookupError`, raised for invalid encoding names like 'nonexistent-enc'). The narrow catch lets `LookupError` propagate as a raw Python exception, bypassing the library's exception hierarchy entirely — across ALL entry points (reader, dev_reader, writer, encoding). Fix: use `except (UnicodeDecodeError, LookupError) as e:` to catch both the expected decode error and the encoding-lookup error. This affected 4 separate call sites in pylasdev-reborn, found independently by 4 different agents.

## [got-20260717164655-aded82]
Category: gotcha
Tags: python, input-validation, guard, negative-value
Changed: 2026-07-17T16:46:55.729376

Negative/zero value bypass after None-default check: `if x is None: x = DEFAULT` followed later by `if x > limit:` silently passes negative and zero values — they're not None (so default never applied) and not above the limit (so guard never fires). An attacker supplies `max_file_size=-1000000` and bypasses the resource exhaustion guard entirely because `is None` is False and the size check only tests for exceeding the (negative) limit. Fix: add `elif x <= 0: raise ValueError(...)` immediately after the None-default check, before any downstream guard uses the value. This is the 2nd PRIOR_FIX_ATTEMPT on this guard in pylasdev-reborn.

## [got-20260718003425-9c2bfd]
Category: gotcha
Tags: python, parser, writer, case-normalization, data-integrity
Changed: 2026-07-18T00:34:25.445450

Case normalization across transform paths: When string values from external input are parsed and compared against internal canonical values, normalize case immediately after extraction. In pylasdev-reborn: parser.py FORMAT_SPEC_PATTERN captured data_format as-is (no .upper()), then 3 comparison sites did case-sensitive checks against uppercase canonical values. Writer.py assumed uppercase section_type (checked via == LOG_DATA) but from_dict input never uppercased it. Fix: apply .upper() once at the extraction site; all comparison sites then naturally match. Verify that ALL code paths touching a parsed string value apply the same case normalization.

## [got-20260718003431-b95121]
Category: gotcha
Tags: python, numpy, ndarray, shared-reference, data-integrity, models, parser
Changed: 2026-07-18T00:34:31.814825

Shared mutable numpy references across object boundaries: When a property returns an internal numpy ndarray as a mutable view (not a copy), all mutations through the property silently propagate to the internal state and vice versa - two objects share one mutable reference with no isolation. In pylasdev-reborn: LASFile.logs property returned DataSection.data ndarray directly; changes through .logs mutated the data section, and changes to the data section mutated .logs. No copy was made. Fix: either return a .copy() to isolate mutations, or return a read-only view and document the aliasing. Prefer copies for public-facing properties that return mutable data; prefer views (for memory) only in internal hot paths with explicit documentation. This is the numpy equivalent of list reference aliasing - mutations are invisible to code inspection.

## [got-20260718003436-244c67]
Category: gotcha
Tags: python, csv, delimiter, quoting, splitting, data-integrity
Changed: 2026-07-18T00:34:36.836161

Delimiter-split without quoting corrupts data with embedded delimiters: str.split(delimiter) has no concept of quoting or escaping - a value containing the delimiter character silently produces extra tokens. In pylasdev-reborn: DLM=COMMA data with values containing commas produced ghost columns; DLM=TAB with tab-containing string data caused column shift. All 5 str.split(delimiter) call sites across parser.py, data_reader.py, dev_reader.py were affected. Fix: use Python csv.reader with delimiter=char and quoting=csv.QUOTE_NONE for LAS-compatible behavior (LAS spec does not define quoting), then strip each token. The csv module handles edge cases that raw split() cannot: trailing delimiters, empty fields, and proper RFC 4180 semantics. When the format spec does not define quoting, use the csv module for robustness anyway - it is more correct than raw str.split() even in QUOTE_NONE mode because it handles field count consistency.

## [got-20260718003439-210944]
Category: gotcha
Tags: python, numpy, allclose, array_equal, ValueError, broadcast, comparison
Changed: 2026-07-18T00:34:39.219527

numpy.allclose and numpy.array_equal raise ValueError for incompatible broadcast shapes: np.allclose(a, b) can raise ValueError "operands could not be broadcast together" when arrays have incompatible shapes - this is distinct from the list-comparison ValueError. In pylasdev-reborn compare.py, np.allclose on a (3,) array vs a (2,3) array raised unhandled ValueError. The existing pattern (try/except ValueError then element-by-element fallback) works but must be applied at EVERY numpy comparison call site, not just list-comparison sites. np.array_equal can also raise ValueError for incompatible shapes. Fix: wrap ALL numpy comparison function calls (allclose, array_equal, array_equiv) in shape validation or try/except. Pre-validate with a.shape == b.shape before calling these functions when shapes are user-controlled.

## [got-20260718003508-6f56a5]
Category: gotcha
Tags: python, numpy, comparison, list, ValueError
Changed: 2026-07-18T00:35:08.190070

numpy array comparison in list containers: v1 != v2 on Python lists that contain numpy arrays raises ValueError (The truth value of an array with more than one element is ambiguous). The list comparison operator broadcasts to element-wise array comparison, which produces boolean arrays, and Python cannot evaluate a boolean array as a single truth value. In pylasdev-reborn _compare_data_sections, the parent compare_las_dicts had a try/except fallback (L134-168) but the list-comparison delegate did not, producing an operational crash. When a parent function has a safety pattern, verify all delegate/helper functions have it too. Fix: wrap list comparison paths in try/except (ValueError, TypeError) with element-by-element ndarray fallback. IMPORTANT: the fallback functions np.array_equal() and np.allclose() can ALSO raise ValueError for incompatible broadcast shapes (e.g., (3,) vs (2,3)). Wrap those calls in their own try/except or pre-validate with a.shape == b.shape. See also got-20260718003439-210944.

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

## [got-20260718050210-313779]
Category: gotcha
Tags: error-handling, exceptions, csv, third-party, pylasdev
Changed: 2026-07-18T05:02:10.225765

Error handling: wrap all third-party library exceptions before module boundaries. Any third-party exception (csv.Error, numpy MemoryError, etc.) must be caught and wrapped in a domain-specific exception (PylasdevError subclass) before escaping the public API. csv.reader() call sites are high-risk — csv.Error occurs on malformed input (unclosed quotes, field-size overflow), exactly the scenario a library should handle gracefully. Raw exceptions propagating to users provide no library context. Search for ALL third-party imports that can raise exceptions not already wrapped — not just csv, but also numpy allocation errors, encoding errors, etc. See findings F-I2-M12, F-I2-M23, F-I2-M04, F-I2-H02 in s5-synth-report.md.

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
Tags: guard, off-by-one, unit-mismatch, dos, pylasdev
Changed: 2026-07-18T22:30:55.082920

Guard operator and unit consistency across code paths: when the same semantic limit is enforced in multiple code paths (parser vs from_dict, pre-scan vs read), both the comparison operator (>= vs >) and the unit of measurement (lines vs characters) must be consistent. >= in one path and > in another creates non-commutative roundtrip behavior — from_dict accepts what the parser rejects. Counting lines in one path and characters in another against the same constant is a unit error. Also: when a guard expression multiplies two counters (e.g., num_curves * actual_count), a zero counter neutralizes the entire guard — every multiplier in a guard expression needs independent minimum validation (> 0). In pylasdev-reborn: parser used >= for 4 MAX guards while from_dict used > for 6 MAX guards (F-I2-M32); MAX_OTHER_LINES checked len(list[str]) in parser but len(str) in from_dict (F-I2-M33); actual_count could be 0, making num_curves * 0 = 0 always pass MAX_TOTAL_ELEMENTS (F-I2-M09).

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

## [pat-20260718223123-34c0a4]
Category: pattern
Tags: testing, falsifiability, guard-coverage, boundary, pylasdev
Changed: 2026-07-18T22:31:23.631238

Test quality — guard coverage and boundary testing: resource limit guards and validation functions with zero test coverage provide false protection. Specific anti-patterns found across pylasdev-reborn: (a) falsifiable-test principle: every test must have at least one assertion that, if the code were broken, would cause the test to FAIL — tests that catch Exception (swallowing AssertionError) or use warnings.warn() as the only failure mechanism can never fail (F-I2-H01, F-I2-M67), (b) guard coverage: every resource limit guard (MAX_CURVES, MAX_TOKENS_PER_LINE, MAX_OTHER_LINES, etc.) must have at least one test that exercises it — guards with zero test references are indistinguishable from missing guards (F-I2-M42, M60, M68, M70), (c) boundary testing: guard tests must cover BOTH sides of the boundary — exactly-at-limit (should pass) AND limit+1 (should fail) — testing only exceed cases leaves off-by-one errors undetected (F-I2-M56, M69), (d) regex variant testing: when a regex has optional components (\s*, \s+), tests must exercise both paths — a path never tested against is a silent regression target (F-I2-M71), (e) null-value testing: tests that set a non-default null value must include at least one data point where it is actually triggered, otherwise the null-resolution path is untested (F-I2-M47, M48), (f) sanitization testing: every sanitization/escaping guard must have a dedicated test that exercises it — an untested sanitizer is indistinguishable from a missing one (F-I2-M49, M51), (g) content verification in roundtrip tests: tests that verify structure (counts, types, section headers) but not content (actual data values) can pass even with data corruption — always include at least one content-level assertion (F-I2-M46, M52, M54, M55).

## [got-20260718223138-e73af5]
Category: gotcha
Tags: normalization, case-sensitivity, reference-data, lookup, pylasdev
Changed: 2026-07-18T22:31:38.744877

Reference data case normalization completeness: when case-insensitive lookup is critical (mnemonic resolution, canonical value matching), ALL entries in the reference database must be normalized to a single case. Even a small fraction of unnormalized entries breaks lookup for that entire subset — quiet failures that are very hard to detect without explicit testing. In pylasdev-reborn: 9 of 1849 canonical mnemonic values (~0.5%) were mixed-case ("Density", "GKst", "NKTst", "Koll") while all other values were uppercase. The parser uppercased lookup KEYS but stored VALUES as-is, so resolution of these 9 entries against uppercase keys silently failed. Checklist: (a) mechanically verify that EVERY value in the reference dataset passes `v == v.upper()` (or the chosen normalization), (b) add a regression test that iterates over all entries and asserts normalization, (c) when a reference dataset is large (1000+ entries), use automated validation — one-off manual fixes will miss entries.

## [pat-20260719022332-c28c46]
Category: pattern
Tags: from-dict, validation, parser, roundtrip, data-model, pylasdev
Changed: 2026-07-19T02:23:32.675803

from_dict validation parity: a from_dict() constructor is the primary bypass vector around parser guarantees. The parser builds objects through a controlled pipeline that enforces key types, cross-field validation, deduplication, and resource limits. from_dict() accepts arbitrary dicts — it must replicate EVERY validation, post-processing step, and guard that the parser normally provides. Specific checklist: (a) validate ALL dict key types before storage (isinstance(key, str) guards) — the parser always produces string keys, from_dict callers may not, (b) cross-validate data keys against curve mnemonics (both string_data and numeric data) — the parser builds these together, from_dict receives them separately, (c) deduplicate curves/sections/entries — the parser calls _deduplicate_curves() but from_dict has no dedup, (d) catch ALL exception types the internal helpers raise (ValueError AND TypeError at every try block, matching sibling classes), (e) apply the SAME resource limits with the SAME operators (>= vs >, lines vs characters) as the parser, (f) detect and handle key collisions across parallel dicts (data vs string_data, metadata vs column names), (g) validate data_sections against supported versions — reject non-3.0 data_sections in from_dict for versions < 3.0 (parser only produces data_sections for LAS 3.0; from_dict must mirror this), (h) cross-validate data_format against actual data placement — accepting data_format='F' with string values causes silent np.float64->np.str_ type change on roundtrip. In pylasdev-reborn: 9 confirmed MEDIUM+ findings across two production-check iterations traced to from_dict bypassing parser guarantees — added findings IF-007 (data_sections version check) and IF-026 (data_format/placement mismatch) in 2026-07 run.

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
Tags: python, writer, sanitization, delimiter, escape, roundtrip, pylasdev
Changed: 2026-07-19T02:23:53.432680

Completeness of field sanitization in delimited output: When a writer uses a sanitization/escaping function for fields that could contain format-significant characters, EVERY field emitted to the output stream MUST use the same function. A single unsanitized field is a silent corruption vector — the omitted field's content can contain characters that the parser interprets as structural delimiters, causing downstream field misrouting and data corruption. In pylasdev-reborn: curve name, unit, and description fields used _escape_colons_for_las_value() but api_code (API code) did not — a colon in api_code caused the parser's colon-separator regex to mis-split the line, routing subsequent field content across wrong columns on re-read. Fix: enumerate all fields emitted by the writer and mechanically verify each one uses the escaping function. A grep for all .format() or string-interpolation call sites in the writer module is the mechanical check. (IF-024 in 2026-07-19 run)

## [got-20260719022356-224371]
Category: gotcha
Tags: python, dict-key, contract, consistency, data-section, pylasdev
Changed: 2026-07-19T02:23:56.591857

Cross-file dict key contract mismatch — 'data' vs 'logs': In pylasdev-reborn, DataSection objects store curve data under the key 'data' (e.g., section['data']), but LASFile-level validation functions accessed it as 'logs' (e.g., data.get('logs')). This key inconsistency caused the per-section data_format validation (_check_df_vs_placement) to silently read nothing from DataSection dicts, making the validation inert for all per-section data. When two types in the same codebase use different dict keys for the same semantic concept, every cross-type function that accesses both is at risk of key mismatch. Fix: define a shared constant (DATA_KEY = 'data') and use it consistently across all types that store or access curve data. Grep for the literal string across all consumers before finalizing the design. (R-001 in 2026-07-19 post-fix review)

## [got-20260719060332-fdcfce]
Category: gotcha
Tags: python, error-handling, try-except, exception-wrapping, structural
Changed: 2026-07-19T06:03:32.473636

Pre-try setup code escapes exception wrapping: Code placed before a try: block is NOT covered by the except clause. When a public API documents that all errors are wrapped in a domain exception, every validation step must be inside the try block. Checklist: (1) identify try block start in every public method, (2) for every line between method signature and try: grep for raise-able calls, (3) move them inside or add wrapping. In pylasdev-reborn: isinstance() and _validate_from_dict_input() ran before the try block in from_dict(), propagating raw TypeError/ValueError. F-008, DOUBLE CONFIRMED (both-found, 2 iterations).

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

## [got-20260719092940-cb116d]
Category: gotcha
Tags: python, lookup, case-insensitive, collision, reference-data
Changed: 2026-07-19T09:29:40.600823

Case-insensitive lookup table key collision detection at build time: When a lookup table loads reference data with mixed-case keys and builds a case-insensitive index (e.g., _mnem_base_upper = {k.upper(): v for k, v in sorted(MNEM_BASE.items())}), multiple original keys can collide to the same uppercased key. In a sorted first-wins construction, the colliding key that sorts first silently shadows all others — downstream lookups resolve to the wrong canonical value with zero warning. Checklist: (a) detect collisions at index build time: compare len(raw_dict) vs len(upper_dict) and warn on difference, or build a collision map {upper_key: [original_keys]} and flag entries mapping to different canonical values, (b) sorted first-wins is the WORST tiebreaker — it's both non-deterministic across Python versions (dict ordering guarantees differ) and data-dependent (adding new entries can silently change resolution order), (c) prefer explicit collision resolution: either reject ambiguous entries at data-build time or maintain both variants with a disambiguation mechanism. In pylasdev-reborn: 'aGK':'GK' (line 1660) and 'Agk':'GRO' (line 1698) both uppercased to 'AGK' — sorted first-wins resolved to 'GRO', making the 'GK' entry dead code through the parser path. Valid mnemonic alias silently resolved to wrong canonical. 1 confirmed MEDIUM+ finding (F-013).

## [got-20260719092951-395cfd]
Category: gotcha
Tags: python, guard, bytes, str, isinstance, type-safety
Changed: 2026-07-19T09:29:51.247832

isinstance(x, str) guards miss bytes — list(bytes_value) produces integers in Python 3: When validation code guards against single-value string inputs with isinstance(curves_order, str), bytes values bypass the guard because isinstance(b'GR', str) is False in Python 3. The downstream list() call on the unguarded bytes value produces integers, not characters: list(b'GR') → [71, 82]. These integers propagate as curve names, column headers, or identifiers, and silently fail downstream lookups against string-keyed dicts — producing empty/null output with zero error. Three characteristics make this dangerous: (a) bytes is a non-list, non-iterable single-value type that list() actually iterates (string-like iteration), (b) the guard was explicitly designed to prevent list(string) → character-by-character corruption, and bytes reproduces EXACTLY the same corruption via a different path, (c) the downstream failure is silent — integer keys match nothing, no KeyError/ValueError is raised, output is simply empty. Checklist: extend str guards to also catch bytes — isinstance(value, (str, bytes)) — at every location that guards against single-value string inputs. In pylasdev-reborn: three guard locations in models.py checked isinstance(curves_order, str) but missed bytes; integer curve names silently produced zero data rows in writer output. 1 confirmed MEDIUM+ finding (F-110).

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

## [got-20260719191500-0d7e9e]
Category: gotcha
Tags: python, exception, try-except, error-handling
Changed: 2026-07-19T19:15:00.018992

Self-defeating exception handler: An except clause that retries the exact same operation without changing any inputs is a guaranteed crash path. The handler catches an exception and re-executes the failing code with the same arguments, same state, same conditions — producing the same exception which propagates as an unhandled error. Checklist: every except block that retries ANY operation must either (a) change the input to make success possible, or (b) wrap in a domain exception and raise. Never retry the identical call. In pylasdev-reborn: dev_reader.py F-I2-DV-05 — IndexError handler at line 954-955 re-executed the same failed numpy write with same slice indices, guaranteeing the same IndexError on the retry. 1 confirmed MEDIUM+ finding.

## [pat-20260719191502-d9b8db]
Category: pattern
Tags: python, validation, type-guard, consistency
Changed: 2026-07-19T19:15:02.573280

Validation guard asymmetry between sibling functions: When two sibling functions perform similar work (e.g., _compare_data_sections and _compare_lists), their validation guards must match. One function with input-type guards and another without creates a crash-only path — data that would trigger the guarded function's catch block hits the unguarded function's unprotected code. Checklist: when auditing sibling/parallel functions, mechanically verify both have equivalent guards. If one has try/except (TypeError, ValueError), the other must too. In pylasdev-reborn: compare.py _compare_lists had type guards; _compare_data_sections did not — F-I2-XCM-01 MEDIUM. Direct construction of mismatched types crashed the unguarded path.

## [got-20260719215427-08a279]
Category: gotcha
Tags: pylasdev-reborn, production-check, sanitization, asymmetry
Changed: 2026-07-19T21:54:27.432993

SPLITLINES_CHARS_RE asymmetry with writer _CONTROL_CHARS_RE: parser/reader/dev_reader strip control chars; writer additionally strips Unicode space chars (\u00A0, \u2000-\u200A, \u202F, \u205F, \u3000). Unicode NBSP from web-pasted content can create phantom section headers. All three SPLITLINES_CHARS_RE instances are identical — fix needed in parser.py:102, reader.py:29, dev_reader.py:28 vs writer.py:45-49.

## [got-20260719215428-5dcab0]
Category: gotcha
Tags: pylasdev-reborn, production-check, guard-consistency
Changed: 2026-07-19T21:54:28.914719

Guard operator inconsistency in parser.py _process_ascii_data: actual_count > MAX_DATA_LINES (line 2411) uses > while num_curves >= MAX_CURVES (line 2416) uses >= in the same function. Operator consistency was previously identified at knowledge.md line 1730. Found in pylasdev-reborn parser.py:2411,2416.

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
Tags: python, save-restore, try-finally, exception-safety, mutable-state, pylasdev
Changed: 2026-07-20T04:03:25.363466

Save/restore patterns must use try/finally: When mutating shared mutable state with save-before/restore-after (e.g., preserving las_file.version.wrap before a write path that overrides it), any exception between the save and restore leaves the state permanently mutated. Bare save/restore without try/finally is exception-unsafe — the restore must be in a finally block. Checklist: (a) enumerate every operation between save and restore — if any can raise, finally is mandatory, (b) the save must happen BEFORE the try block, (c) the restore must be the ONLY code in the finally block (it must not itself raise). Defensive pattern: even when all operations appear infallible, use try/finally anyway — future code additions between save and restore should not require remembering to add exception safety. In pylasdev-reborn: writer.py R8-007 data_sections copy-back mutated model state without save/restore; G-018 version.wrap mutation without restore; S10-001 post-fix found exception-safety gap in the added save/restore (no finally). 3 CONFIRMED findings across 3 fix convergence passes.

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

## [pat-20260720040342-cf5e44]
Category: pattern
Tags: testing, assertion, regression, false-positive, quality, pylasdev
Changed: 2026-07-20T04:03:42.784810

Weak test assertions produce false confidence — assert len>0 and assert key-in-dict without verifying the specific value: When a test asserts that a parsed value is non-empty (assert len(las.well['COMP']) > 0) or present (assert 'COMP' in las.well), both pre-fix (broken) and post-fix (correct) code produce non-empty, present values. The assertion provides ZERO regression value — it passes identically before and after the fix. Every production-fix regression test must assert the SPECIFIC expected value or property the fix changes. Checklist: (a) the assertion must FAIL on pre-fix code and PASS on post-fix code — if it passes on both, it's a no-op, (b) for string sanitization fixes: assert '\u00A0' not in result (checking absence of the specific problematic character), (c) for value semantics: assert result == 'expected specific value'. In pylasdev-reborn: F-001 NBSP test asserted only key presence + non-empty; G-018 wrap test used default 'NO' that never triggered the mutation path. 3 CONFIRMED MEDIUM findings (R8-006, S10-004, S10-005) — weak test assertions survived adversarial review and required a second fix convergence pass.

## [got-20260720040350-f527a9]
Category: gotcha
Tags: testing, default-value, no-op, regression, quality, pylasdev
Changed: 2026-07-20T04:03:50.975455

Default values in non-default tests produce no-op tests: When testing that a non-default code path works (e.g., verifying that write operations restore mutated state), using the default input value (e.g., wrap='NO') often bypasses ALL mutation branches. The writer only mutates wrap when it differs from 'NO' — using the default means the save/restore is a mechanical no-op, and the test passes identically whether restore exists or not. Fix: use a non-default input that ACTUALLY triggers the mutation path (e.g., wrap='YES' which the writer overrides to 'NO' for LAS 1.2 WRAP=YES files). Then assert the pre-write original value is preserved after writing. Checklist: (a) identify which branch the test should exercise — the DEFAULT branch or the non-default branch, (b) if non-default, verify the input value is genuinely non-default by checking the model's field defaults (models.py:494 for version.wrap), (c) verify the mutation path actually fires: grep the write function for conditions that check this field and confirm the test input would trigger them. In pylasdev-reborn: test_wrap_non_default_restored set wrap='NO' (default per models.py:494); writer mutation at line 375 only fires for 'YES', line 393 only for non-'NO' LAS 3.0 — both skipped. 1 CONFIRMED MEDIUM finding (S10-005).

## [pat-20260720040355-cdc0a8]
Category: pattern
Tags: python, write, serialize, mutation, save-restore, side-effect, pylasdev
Changed: 2026-07-20T04:03:55.182686

Write/serialize operations should not permanently mutate the input model: Functions that modify shared mutable objects for internal computation (e.g., copy-back of data_sections contents to las_file for writing, overriding version.wrap for format compliance) must restore the original state before returning. The input model may be reused after the write (double-write scenario: first write mutates logs, second write sees stale state and skips proper data); consumers may inspect model state post-write. Checklist: (a) save each attribute before mutation — deep copy if nested, shallow copy if value types, (b) restore all saved attributes after the write operation completes, (c) for complex mutation blocks with potential early returns or exceptions: use try/finally to guarantee restore, (d) verify model state is unchanged after write: assert original_wrap == las_file.version.wrap after write. In pylasdev-reborn: writer.py _write_version_section mutated version.wrap (G-018); _write_ascii_sections mutated logs/string_data/curves_order/curves without save/restore (R8-007). 2 CONFIRMED MEDIUM+ findings, both regression hotspot files (writer.py has 3+ PRIOR_FIX_ATTEMPT findings).

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
Tags: pylasdev, parallel-code-path, fix, regression, audit
Changed: 2026-07-20T07:18:38.851639

Parallel code path omission — when a fix, validation, or branch is applied to one code path, mechanically verify ALL sibling paths received the same change: When a bug class is identified and fixed in one location, the same bug propagates through every structurally-similar code path. This is the #1 recurring production-check pattern — across this run alone: (a) tab-in-mnemonic validation added to CurveDefinition (F-013) but missed in ParameterEntry (F-031) — same prior fix commit de3aa48 missed both, (b) data_format truncation write-back applied to top-level curves loop (F-017) but missed in per-section loop (F-R01) — identical two-phase code structure, (c) dict-with-ndarray recursion branch added to compare_las_dicts inner loop (F-028) but missed in _compare_data_sections inner loop (F-R02) — sibling functions at same call depth, (d) generator materialization applied to top-level curves_order (F-029) but missed in per-section curves_order, (e) _failure_counter parameter passed to one _to_finite_float call site but missed at 4 other call sites in same function (F-035). Checklist: after every fix or validation addition, grep for structurally-similar code in sibling functions, per-section/branch siblings, or parallel call sites — the bug class is unlikely to be limited to the location you found it. Fork-in-code (if/elif/try branches) and for-section loops are the highest-risk structures for mechanically missed parallel paths.

## [got-20260720071845-a030ba]
Category: gotcha
Tags: pylasdev, exception, try-except, regression, incomplete-prior-fix
Changed: 2026-07-20T07:18:45.655200

Incomplete except clause after code change: when a  is added inside a  block, the downstream  clauses must be updated to catch it. F-026 at data_reader.py:205-210 is the canonical example: prior fix 196c78b8 added  for non-finite NULL sentinel values inside the try block, but left  unchanged. LASParseError inherits from PylasdevError → Exception — NOT ValueError or TypeError. The raise escaped the except handler at all 4 call sites. Checklist: (a) when adding a  inside try, grep for the matching except clause and verify SomeError or its superclass is in the exception tuple, (b) verify ALL outer wrapper except clauses that re-raise as domain exceptions also catch the new class, (c) actively generate an input that triggers the new raise path and confirm the except clause catches it — if the exception propagates uncaught, the except clause is incomplete.

## [pat-20260720071904-22bb7b]
Category: pattern
Tags: pylasdev, roundtrip, writer, parser, symmetry, format
Changed: 2026-07-20T07:19:04.467128

Roundtrip format symmetry — writer output tokens must match parser input tokens: When a writer emits tokens (format specifiers, delimiters, quoting) that the parser does not consume, or the parser consumes tokens the writer does not emit, every roundtrip is silently lossy. Two variants from this run: (a) F-004: csv.reader(QUOTE_MINIMAL) consumed double-quote characters as CSV quoting, stripping them from values on parse. Writer used raw delimiter.join() with no quoting — the stripped quotes were never re-emitted, so quoted strings permanently lost their quotes on roundtrip. Fix: use str.split(delimiter) — no quoting interpretation — matching the writer. (b) F-016: writer emitted :offset suffix for all data_format values, but parser at parser.py:1925 only processed offset when data_format=="A". Non-"A" offsets were silently discarded on roundtrip. Fix: restrict writer emission to data_format=="A" only. Checklist: for every token the writer emits, verify the parser has a code path that consumes it and produces the same meaning. For every token the parser expects, verify the writer emits it unconditionally when the parser requires it.

## [got-20260720071943-5e0df7]
Category: gotcha
Tags: pylasdev, validation, None, empty-string, falsy, guard
Changed: 2026-07-20T07:19:43.403584

Falsy truthiness check (if x:) conflates None (invalid) with empty string (valid) in model validation guards: F-038 (models.py:523) used 'if self.dlm:' to gate validation, but None and '' are both falsy, so None silently passes. Downstream code (delimiter_char .upper()) then raises AttributeError on None. F-010 (writer.py:413) used 'if dlm.upper() != SPACE' which correctly handles '' but never guards against None. When a field has three states (None=invalid, ''=not-specified, 'COMMA'=specified), 'if self.dlm:' dispatches incorrectly for None. Fix: use 'if self.dlm is not None:' to reject None while accepting ''. This is distinct from the input-validation variant (got-20260717204121-2947ea) — that's about dict deserialization truthiness dispatch; this is about model-level guard semantics.

## [got-20260720071943-cbbcd4]
Category: gotcha
Tags: pylasdev, generator, iterable, materialize, validation, data-loss
Changed: 2026-07-20T07:19:43.477519

Generator exhaustion in multi-pass consumption: when an iterable (especially a generator) is validated in a for-loop and then a second loop or list comprehension re-iterates it, the second pass gets zero elements — silently producing empty output. F-029: curves_order from from_dict was validated via enumerate(curves_order) loop, then passed to [c for c in curves_order] which got zero elements when it was a generator. Fix: materialize to list() immediately after type check. Checklist: after every isinstance(x, Iterable) check, add x = list(x). Grep for any function that iterates the same parameter twice — second loop consuming zero is silent corruption.

## [got-20260720071943-a34bd7]
Category: gotcha
Tags: pylasdev, normalization, canonicalization, collision, from-dict, validation
Changed: 2026-07-20T07:19:43.554991

Normalization/canonicalization collision detection: when two distinct raw keys normalize to the same canonical key, the normalization MUST detect and reject the collision rather than silently overwriting. F-030 (models.py:2466-2470): DevFile.from_dict normalization mapped column aliases to canonical names (MDKB→MD, md→MD) but if input already had key 'MD' AND key 'MDKB', both normalize to 'MD' — second silently overwrites first. Fix: before data[norm_key] = data.pop(raw_key), check 'if norm_key in data: raise ValueError' with both key names. Checklist: every normalization/alias map must check for collisions between normalized keys and pre-existing keys, and between two raw keys that normalize to the same target.

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
Tags: python, exception, api-contract, module-boundary, pylasdev
Changed: 2026-07-20T10:46:19.306102

Exception type consistency at module boundaries: Every raise site within a module's public API must use the exception type documented for that module's contract. Using a general exception from a different domain (e.g., LASDataError in a parser function that documents LASParseError, or LASDataError in a DEV reader that documents DEVReadError) breaks caller exception handling — the caller catches the documented type expecting to handle module-specific errors, but the actual exception escapes uncaught. F-01: parser.py raised LASDataError at line 1682 (should be LASParseError) and RuntimeError at line 2668 (should be LASParseError) — both violated the parser module's documented exception contract. F2-20: dev_reader.py raised LASDataError at line 1012 while the module's 22 other raise sites used DEVReadError — a single inconsistent raise site in a documented API. Both caused miss-caught exceptions at caller catch blocks. Fix: (a) define a module-specific base exception (e.g., ParserError, DevReaderError), (b) all raise sites in the module's public functions use only that exception or its subclasses, (c) grep for raise in the module and verify every raise site uses the module-approved type — a single outlier is a contract violation. Checklist: in every module, grep for ^\s*raise\s+\w+Error — every match should be the module's documented exception type. Exception types from other modules (LASDataError in parser code, RuntimeError anywhere) are contract violations.

## [got-20260720104633-378dcc]
Category: gotcha
Tags: python, writer, serialize, model-state, compliance, pylasdev
Changed: 2026-07-20T10:46:33.670890

Writer model must reflect disk truth for compliance-overridden fields — don't restore pre-write values that the writer deliberately changed: When a writer overrides a model field for format compliance during serialization (e.g., forcing wrap='NO' for LAS 1.2 WRAP=YES files), the model state after write must reflect what was actually written to disk — NOT the pre-write value. Restoring the compliance-overridden field to its pre-write value creates a model-disk contradiction: the model claims one value (wrap='YES') but the file contains another (WRAP=NO). F2-26 (writer.py:366-425): the writer correctly set las_file.version.wrap = 'NO' at line 389 to match what was written, but the finally block at line 425 unconditionally restored original_wrap — for wrap='YES' input, model claims YES but file says NO. Compounding: to_dict() returns {'WRAP': 'YES'} but the actual file is WRAP=NO. For wrap=None: model restored to None, to_dict() violates dict[str,str] contract. This is the inverse of pat-20260720040355-cdc0a8 (which correctly advises saving/restoring internal computation state): for compliance-overridden fields the writer intentionally changes, DO NOT restore — the model must be honest about what was written. Checklist: (a) identify which model mutations during write are internal computation (restore these) vs. compliance overrides (do NOT restore these), (b) if a field has two purposes, flag the conflict — an override+restore pair is always a bug (either the override is correct and restore is wrong, or vice versa), (c) after write, assert las_file.version.wrap == actual_written_value — verify model-disk consistency.

## [got-20260720203918-eae414]
Category: gotcha
Tags: python, copy, mutation, defensive-copy, shallow-copy, deepcopy, pylasdev
Changed: 2026-07-20T20:39:18.631307

Shallow copy (.copy()) leaks nested dict mutations to callers: When a method returns a shallow copy of internal state containing nested mutable values (e.g., data = self._curves.copy()), the returned dict's nested dicts are shared references — anything the caller does to those nested dicts silently mutates the object's internal state. The function intended to protect itself from caller mutations but shallow copy only protects the top level. Fix: use copy.deepcopy() when the dict contains nested mutable values (dict values, list values with mutable elements). In pylasdev-reborn: F-002 (HIGH, both-found) at models.py:1464 — models() returned a shallow copy of '_curves' dict, but the nested DataSection.values dict was a shared reference. Caller mutations to returned data leaked directly into the model's internal state. Checklist: (a) grep for .copy() in return statements and constructor getter methods, (b) verify returned dicts have no nested mutable values — if they do, use deepcopy, (c) alternatively: use immutable types (tuple of NamedTuples, frozenset) for nested values to make shallow copy safe, (d) document the copy semantics in the docstring — shallow vs deep is a contract decision.

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
Tags: python, type-checking, dict, iterable, isinstance, guard
Changed: 2026-07-21T04:35:17.334249

Python dict passes isinstance(x, Iterable) — dict iteration yields keys, not values: In Python, dict is Iterable (collections.abc.Iterable). isinstance(my_dict, Iterable) returns True. When a guard chain checks None → str/bytes (single-value) → Iterable (multi-value) to distinguish single values from sequences, a dict passes all three guards: it's not None, it's not str/bytes, and it IS Iterable. The downstream iteration then silently processes dict KEYS instead of values — if the code expected a list of mnemonics like ['GR', 'DT'], it instead receives ['GR', 'DT'] from dict keys (same result, accidentally correct) OR receives integer keys silently converted to strings, producing wrong-but-valid-looking output with zero error. This is the same class as bytes-passing-Iterable (got-20260719092951-395cfd) but with a different trap: dict IS genuinely iterable (unlike bytes which iterates to integers), so the guard is not wrong — it's incomplete. The Iterable guard must ADDITIONALLY reject dict: add 'isinstance(curves_order, dict)' BEFORE the Iterable check. Checklist: (a) for every isinstance(*, Iterable) guard on user input, mechanically verify dict is rejected before the Iterable branch, (b) test with dict input — if no TypeError is raised, the guard is missing, (c) prefer explicit type validation: accept list, reject dict — don't rely on Iterable as a catch-all. In pylasdev-reborn: models.py:1734 guard chain accepted dict as curves_order; dict keys silently used as curve names. Confirmed MEDIUM (F2-014).

## [pat-20260721043524-c318eb]
Category: pattern
Tags: python, defensive-programming, cap-bypass, min, max
Changed: 2026-07-21T04:35:24.439037

Sequential cap cancellation — min(x, CAP) undone by downstream max(x, uncapped): When a value is capped with min(value, CAP) and then a downstream operation takes max(value, other_value), the cap is structurally negated if other_value can exceed CAP. Pattern: 'capped = min(raw, 100)' followed later by 'result = max(capped, other)' — if 'other > 100', result exceeds the cap. The max() undoes the min() cap. This is an emergent property — neither function is wrong in isolation, the bug is in the ordering and the assumption that a downstream max() will not re-exceed the cap. Fix: apply the cap at the FINAL computation site (after all arithmetic), or cap the downstream input too: 'other = min(other, CAP)'. Alternative: cap after max: 'result = min(max(capped, other), CAP)'. Checklist: (a) trace every capped value through all downstream arithmetic — any operation that can increase the value beyond CAP is a bypass, (b) max(), addition, multiplication, and exponentiation are the most common bypass vectors, (c) prefer single-point capping: apply min(..., CAP) as the LAST operation before use, not the first. In pylasdev-reborn: writer.py:1419 capped decimal_places = min(raw, 100), but line 1434 used max(decimal_places, sig_digits) where sig_digits had no cap → when sig_digits > 100, result exceeded cap. Confirmed MEDIUM (F2-016).

## [pat-20260721043532-309dca]
Category: pattern
Tags: python, resource-guard, maxsplit, split, parsing
Changed: 2026-07-21T04:35:32.653994

Unbounded str.split() without maxsplit on potentially-large input — resource exhaustion: When string splitting is applied to lines from an untrusted file, a line with N delimiters produces N+1 tokens — all allocated as a Python list. Without maxsplit, a single pathological line (millions of commas) can allocate a list with millions of string elements from one line of input. File-size limits provide only indirect mitigation (a 50KB line with comma every 2 chars = 25K tokens — manageable, but a 500KB line = 250K tokens which is material). The fix is mechanical: add maxsplit=MAX_TOKENS to every str.split() call in parsing/detection code. This is distinct from CSV quoting issues (got-20260718003436-244c67) — even with csv.QUOTE_NONE, csv.reader still allocates field lists proportional to delimiter count. For format-detection code that splits header/column lines to inspect structure, use str.split(delim, maxsplit=N) where N is bounded by MAX_CURVES or a similar domain limit. Checklist: (a) grep for '\.split(' and '\.splitlines' in parser/reader code, (b) for every split on file content: verify maxsplit is set to a domain-appropriate bound, (c) the 3 highest-risk patterns are: delimiter auto-detection (splits many test lines), format header parsing (splits column-name lines), and data line counting (splits every data line). In pylasdev-reborn: 10 of 13 split() calls in dev_reader.py had maxsplit; 3 comma-splits in format detection did not. Confirmed MEDIUM (F2-021).

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
Tags: python, pylasdev, testing, regression, coverage
Changed: 2026-07-21T18:06:22.663593

Every bugfix must include regression tests covering the fixed code path: When a bug is fixed in comparison functions, writer logic, or model validation, regression tests must explicitly exercise the exact code path that was broken. A fix without a corresponding test is indistinguishable from unbroken code — future refactoring can re-introduce the same bug with zero test failures. Specific requirements: (a) the test must FAIL on pre-fix code and PASS on post-fix code — a test that passes on both is not a regression test, (b) for comparison function fixes: test with the exact data structure that triggered the bug (e.g., nested dict with list-of-ndarray values), (c) for state mutation fixes (WRAP, DLM save/restore): assert the post-write model state matches what was written to disk, not what the model was before writing, (d) if a prior fix already has a test class for the same feature (e.g., TestG018WrapPreservation for WRAP), the new fix's test should mirror that pattern. Checklist: after every fix, verify: (1) does a test exist that specifically exercises the fixed lines? (2) does the test fail when the fix is reverted? (3) does the test assert the specific expected value, not just presence/non-emptiness? In pylasdev-reborn: s8 post-implementation review found M-01 (no regression test for list dispatch comparison branches in compare.py) and M-10 (zero DLM post-write state assertions despite 62 DLM references in test_writer.py — all execution-coverage, zero assertion-coverage). Both MEDIUM.

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
Tags: python, unicode, whitespace, sanitization, silent-mutation, pylasdev
Changed: 2026-07-22T03:56:28.595921

Unicode whitespace characters silently deleted by control-character regexes: Regexes targeting ASCII control characters ([\x00-\x08\x0b\x0c\x0e-\x1f\x7f]) as 'characters to strip' silently delete Unicode whitespace characters that fall outside this range — NBSP (\u00A0), en/em/hair spaces (\u2000-\u200A), narrow NBSP (\u202F), math space (\u205F), and ideographic space (\u3000). These are legitimate whitespace characters that should be replaced with spaces, not silently deleted, because deleting them concatenates adjacent tokens (e.g., 'ABC\u00A0DEF' becomes 'ABCDEF' instead of 'ABC DEF'). The regex must be Unicode-aware: either use Python's str.isprintable() or explicitly enumerate Unicode whitespace categories to replace them with ' ' rather than strip them. Fix: separate 'characters to delete' (true control chars: \x00-\x1f excluding \t\n\r, plus \x7f) from 'characters to replace with space' (Unicode whitespace: \u00A0, \u2000-\u200A, \u202F, \u205F, \u3000). Checklist: (a) in every sanitization regex, check if Unicode whitespace chars match, (b) verify they are replaced with space (' '), not deleted, (c) test with input containing NBSP and en-space — verify tokens separated by them remain separated after sanitization. In pylasdev-reborn: _writer_base.py _CONTROL_CHARS_RE silently deleted \u00A0 and \u2000-\u200A chars — concatenating adjacent well-log values. 1 confirmed HIGH finding (F-05).

## [got-20260722035635-f9576c]
Category: gotcha
Tags: python, concurrency, module-level, shared-state, thread-safety, pylasdev
Changed: 2026-07-22T03:56:35.413290

Module-level mutable flags as per-instance configuration: A module-level variable used as a configuration toggle (e.g., _DESANITIZE_ENABLED) creates a race condition when multiple callers set different values concurrently. Two threads both calling read_las_file() with different desanitize parameters will race on the single module-level flag — the second caller's value overwrites the first's before the first finishes, causing the first to process data with the wrong sanitization setting. This is invisible in single-threaded tests because only one caller exists at a time, but breaks under any concurrent use (web server, thread pool). Fix: thread the configuration through function parameters or per-instance state instead of module-level globals. Preference order: (1) pass as parameter to every function that needs it, (2) store on a per-parser-instance attribute (self._desanitize) not on the module, (3) if a module-level constant is truly needed, use threading.local() or contextvars.ContextVar. Checklist: (a) grep for module-level variables set inside functions (assignment in function body to global/module-level name), (b) verify each is either read-only constant or thread-safe, (c) for any variable that changes per-call, trace all concurrent call paths — if two threads can set different values, it's a race. In pylasdev-reborn: parser.py _DESANITIZE_ENABLED was set to True/False per read_las_file call at line 501, but all parsing functions read it from module level — concurrent callers with different desanitize values race. 1 confirmed MEDIUM finding (F-21).

## [got-20260722035642-bcb034]
Category: gotcha
Tags: python, version, hardcoded, backwards-compatibility, pylasdev, las-spec
Changed: 2026-07-22T03:56:42.077237

Hardcoded version-specific constants when version-aware data structures exist: When a codebase defines version-specific configuration (e.g., _LASVersionSpec.mandatory_well_fields with per-version well-field requirements), code paths that hardcode a single version's constants instead of consulting the version-aware structure silently apply wrong-version rules. The parser hardcoded 8 mandatory well fields (the LAS 1.2 set) and applied them to ALL versions — LAS 2.0 files got warnings about missing 1.2-specific fields they legitimately don't require, and LAS 3.0 files with 1.2 fields satisfied the check despite 3.0 requiring different fields. The version-aware data structure existed in the same file but was never called. This is a documentation/codebase-awareness failure: the infrastructure for correct behavior was built but the consumer code was never updated to use it. Fix: mechanically grep for all usages of the hardcoded constant and replace with the version-aware lookup. Checklist: (a) when adding version-aware configuration, grep for all existing hardcoded references to the same constants in the same module, (b) each hardcoded reference is a bug waiting to happen — either remove it or redirect through the version-aware lookup, (c) verify behavior change with test cases for every version. In pylasdev-reborn: parser.py:613-616 hardcodes ['COMP', 'WELL', 'FLD', 'LOC', 'PROV', 'CNTY', 'STAT', 'CTRY'] for all versions; _LASVersionSpec in models.py:255 provides correct per-version mandatory fields. 1 confirmed MEDIUM finding (F-25).

## [got-20260722035649-3a7edc]
Category: gotcha
Tags: python, testing, false-pass, filter-mismatch, masking, pylasdev
Changed: 2026-07-22T03:56:49.433460

Test string-filter mismatch producing false passes: When a test filters for a specific warning/error message string that does not match what the code actually emits, the test passes with zero warnings detected — masking the real failure. The test appears to cover the behavior but actually covers nothing because the filter silently drops all actual output. This is mechanically similar to an empty catch block: the test's assertion fires against an empty list, which always passes. The mismatch often occurs when warning messages are changed in a code fix but the corresponding test filter string was never updated, or when the test was written against an incorrect understanding of the output format. Fix: (a) the test filter string must be an EXACT substring of the actual emitted warning, (b) after any change to warning/error message text, grep for the old text in all test files and update filter strings, (c) add a secondary assertion: assert len(filtered_warnings) > 0 — verify at least one warning was captured before asserting on its content. A zero-length capture should fail immediately rather than pass silently. Checklist: (a) for every warnings.catch_warnings block with substring filtering, verify the filter matches actual output strings, (b) add a post-filter assertion that count > 0, (c) when refactoring warning messages, mechanically grep test files for old text strings. In pylasdev-reborn: test_parser.py:1524-1547 filtered for 'LAS 2.0 file missing mandatory well field' but parser actually emitted 'LAS 1.2 file missing mandatory well field: WELL' — test passed with 0 warnings detected despite 4 actual warnings being emitted. 1 confirmed MEDIUM finding (I2F-11).

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

## [con-20260722035851-66a5cc]
Category: context
Tags: pylasdev, production-check, complete
Changed: 2026-07-22T03:58:51.290355

pylasdev-reborn production audit COMPLETE (2026-07-22). 136 agents, 9 stages. 55 MEDIUM+ findings fixed. 838 tests pass, lint+typecheck clean. 8 new knowledge patterns harvested. 14 files changed (+955/-249).

## [got-20260723180811-fa01d1]
Category: gotcha
Tags: python, pylasdev, mutation, validation, containers, __setitem__
Changed: 2026-07-23T18:08:11.527819

Container mutation guards for collection wrappers: When a class wraps a mutable container (dict, list) and exposes it via __setitem__ or public attributes, every mutation bypasses type/value validation — the raw container accepts anything. Post-hoc validate() catches some issues but cannot prevent invalid state from existing. Fix: enforce type/value constraints at every mutation entry point, not just at validate() time. In pylasdev-reborn: WellSection.__setitem__ accepted non-string keys (F-002) and non-str values without coercion (F-006); DataSection.data/string_data exposed as plain dicts with zero mutation guards (F-008); LASFile.logs/string_data, curves, parameters exposed as plain dicts/lists with zero guards (F-009, F-030) — 5 confirmed MEDIUM findings across 3 classes.

## [got-20260723180813-2fe99a]
Category: gotcha
Tags: python, pylasdev, dataclass, validation, post_init, construction
Changed: 2026-07-23T18:08:13.819528

Silent empty-value acceptance in __post_init__: When a dataclass's __post_init__ accepts empty or invalid values for fields that from_dict/parser would reject, direct construction (MyClass(field=None)) silently produces an invalid object. Since __post_init__ runs for ALL construction paths, it's the universal gate — if it doesn't validate, no path validates. Fix: __post_init__ must apply the SAME mandatory-field checks as from_dict/parser, or delegate to a shared validation function. In pylasdev-reborn: Empty VERS passed silently through __post_init__/validate (F-005); LASFile.__post_init__ had zero mandatory well field validation — from_dict and parser checked but direct construction had none (F-036, F-067). 3 confirmed MEDIUM findings.

## [got-20260723180816-b4fbc3]
Category: gotcha
Tags: python, pylasdev, auto-detection, parameters, asymmetric, validation
Changed: 2026-07-23T18:08:16.533020

Auto-detection validation bypassed by explicit parameters: When auto-detection logic (e.g., delimiter inference from file content) performs data-quality checks or corrections, code paths that accept explicit parameters skip those checks — the user who specifies a value gets silently worse behavior than the user who lets the library auto-detect. Fix: run the same validation/correction logic regardless of whether the parameter was auto-detected or explicitly provided. Checklist: (a) for every function with auto-detected defaults, grep for validation/correction that runs only in the auto-detect branch, (b) extract the validation into a shared function called by both paths. In pylasdev-reborn: Delimiter auto-correction and cross-validation only ran when delimiter=None; explicit wrong delimiter produced silent data corruption (F-013).

## [got-20260723180819-1fab4b]
Category: gotcha
Tags: python, pylasdev, version, compatibility, dispatch, routing
Changed: 2026-07-23T18:08:19.481176

Catch-all else branch as version dispatch for future compatibility: When a version dispatch routes unknown/future versions to a specific version's handler via else or default branch, silently applying wrong-version semantics. The system should either reject unknown versions with a clear error, or at minimum warn that an unrecognized version is being treated as version X. Fix: replace catch-all with explicit version enumeration; add unknown-version rejection or warning. Checklist: (a) grep for else/default branches in version-routing dispatch tables, (b) verify each has either an explicit unknown-version handler or reject-path, (c) never silently route unrecognized versions to an existing handler. In pylasdev-reborn: _writer_base.py version dispatch used else→_Las30Writer as catch-all for any version; no version validation in write path (F-020).

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

## [got-20260723180830-c58e2a]
Category: gotcha
Tags: python, pylasdev, normalization, construction, from_dict, parser
Changed: 2026-07-23T18:08:30.482333

Inconsistent preprocessing between construction paths: When multiple code paths construct the same object from raw input (e.g., from_dict vs parser), but apply different preprocessing/normalization (case folding, key normalization, value coercion), the same logical input produces different internal state depending on which path was used. Fix: extract shared preprocessing into a single function called by all construction paths. Checklist: (a) enumerate all paths that construct the same object type, (b) verify each applies identical normalization, (c) test with the same input through each path and assert identical output. In pylasdev-reborn: from_dict() did not normalize well entry keys through mnem_base, but the parser did — behavioral inconsistency (F-058). Related: F-012 where DevFile.from_dict() skipped _validate_dev_data() that other construction paths performed.

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

## [pat-20260801161344-fc653b]
Category: pattern
Tags: python, wrap-detection, heuristics, regression, reader
Changed: 2026-08-01T16:13:44.198585

Format/wrap detection heuristics must corroborate over multiple lines AND be curve_count-aware. Single/first-line-only checks are defeated by: two consecutive sparse rows (D-02), genuine WRAP=YES with overfull second line (D-03), wrapped COMMA/TAB files with >=3 curves (EXT-01 — a REGRESSION from the D-01/D-02/D-03 protocol rewrite: pre-fix comma depth-line heuristic len<=1 handled it, rewrite's full_count>=2 majority vote misdetects). D-01: comma/tab path lost F-M16 corroboration. N-I-08: wrapped-mode under-fill guard must cover mid-file steps, not just trailing EOF. P-15: single-curve WRAP=YES-before-VERS needs post-VERS re-check. Checklist: corroboration must be curve_count-aware (blanket >=2 values breaks 2-curve wrapped files); protocol-based detection over >=3 lines; a detection rewrite must keep previously-correct input classes passing (regression-test them). In pylasdev-reborn: D-01(D-03, EXT-01, EXT-05, N-I-08, P-15 — 7 CONFIRMED (1 HIGH regression EXT-01).

## [pat-20260801161434-05fd52]
Category: pattern
Tags: python, validation, whitelist, parser, roundtrip
Changed: 2026-08-01T16:14:34.604521

Identifiers and units must be validated against a WHITELIST matching the parser grammar, not a punctuation blacklist — blacklists silently miss novel characters. M-03: ALL punctuation (#, |, ;, (, @, $, ,, /, \, +, =, ~, [abc]) silently drops curves, not just colons — whitelist against parser grammar ^\w[\w\-]*(\[\d+\])?$. M-04/N-I-22: unit char class [\w\-/]* rejects %/degC/ohm.m → whole curve + data column silently dropped (HIGH, data irrecoverable). M-27: whitespace section_type misrouted to ~O — unify sentinel, reject spaces AND |. N-I-19: well keys with dots/spaces/colons ENTIRELY DROPPED on re-read. N-I-07: unknown single-char format {X} — parser warns-and-clears vs from_dict RAISES, roundtrip crashes — align gates. Checklist: every identifier field (curve mnemonic, unit, well key, section_type, data_format) needs a whitelist regex matching what the parser accepts; validate on BOTH model layer and parser layer (CurveDefinition AND ParameterEntry). In pylasdev-reborn: M-03, M-04, N-I-22(HIGH), M-27, N-I-19, N-I-07 — 6 CONFIRMED.

## [pat-20260801161434-5a047d]
Category: pattern
Tags: python, roundtrip, writer, parser, symmetry, format
Changed: 2026-08-01T16:14:34.696199

Roundtrip writer/parser token symmetry: when a writer emits tokens (format specifiers, brackets, escapes, section targets) the parser must consume them identically, and vice versa. N-I-18: writer appends {F} at END but parser takes FIRST FORMAT_SPEC_PATTERN match — user text {F} stripped, data_format mis-extracted. M-23/N-I-21: parameter/curve data_format multi-char divergence — parser clears, from_dict truncates to first char, writer emits unbraced — one-time duplication on roundtrip. W-08/W-09: array_index/array_info dropped without bracket mnemonic → parameter/curve reclassified to string_data; bracket emission needed (CAUTION double-bracketing). N-I-02: ZONE_ASSOC_PATTERN runs unconditionally (no version gate), writer never re-emits zone for non-3.0 → text lost. Checklist: for every writer-emitted token verify parser consumes it and vice versa; version-gate version-specific patterns; bracket mnemonics must align data dict keys/curves_order. In pylasdev-reborn: N-I-18, M-23, N-I-21, W-08, W-09, N-I-02 — 6 CONFIRMED.

## [pat-20260801161434-098911]
Category: pattern
Tags: python, container-guard, mutation, writer, validation
Changed: 2026-08-01T16:14:34.877738

Guard wrappers around mutable containers must be re-established after write and cover ALL mutation entry points. W-06: writer finally blocks strip _GuardedDict/_GuardedList → guards PERMANENTLY disabled after any write (M-01/M-02 fixes incomplete without re-wrap). W-07: failure-path wrap leak + 6 dead _saved_* fields — success-flag restore needed. N-I-14: _DevColumns (and _GuardedDict) must override update/pop/setdefault/clear — C-level dict methods bypass __setitem__/__delitem__ (verified empirically) → columns/column_order desync → to_dict/from_dict LASDataError. M-02: _GuardedList init bypass + slice rejection. M-01: int logs keys silently dropped, crash via curves_order/well.entries. Checklist: grep for finally blocks that unwrap guarded containers — re-wrap required; override ALL C-level dict methods; guard every construction path including __init__ and slicing. In pylasdev-reborn: W-06, W-07, N-I-14, M-02, M-01 — 5 CONFIRMED.

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

## [pat-20260801161435-e13efb]
Category: pattern
Tags: python, encoding, cyrillic, detection, mojibake
Changed: 2026-08-01T16:14:35.515922

Encoding detection must sample enough bytes and score by ratio, and Cyrillic tiebreaks must be byte-precise. E-07: byte-frequency AND run-length samples limited to first 64K → Cyrillic beyond 64K → mojibake; widen BOTH samples. E-06: cp1251 + '№' (0xB9) → cp1252 mojibake — naive preferred-encoding-primary sort fix would REGRESS UTF-8 Cyrillic; ratio-primary sort is load-bearing, target the specific 0xB9 per-char advantage (2.9% ratio gap). E-03 (LOW): UTF-16/32 BOM tiny-file branch falls to cp1251. Checklist: sample the whole file or size-proportional window; never switch the primary sort without checking the dominant-encoding regression; per-char byte signatures beat 1-char alnum advantages. In pylasdev-reborn: E-07, E-06 — 2 CONFIRMED.

## [pat-20260801161518-ecfb93]
Category: pattern
Tags: python, writer, dedup, per-section, roundtrip, parser
Changed: 2026-08-01T16:15:18.966679

Writer-side cross-section curve dedup must be per-section scoped, dedup keys must include distinguishing attributes, and the fix must land on BOTH writer and parser. W-01 (HIGH): duplicate curve emission — dedup alone INSUFFICIENT: per-section curve scoping required; dedup-by-mnemonic silently drops differing definitions (DEPT.M vs DEPT.FT); root cause spans writer AND parser (parser registers curves from top-level ~C AND each ~_Definition — writer-only dedup still inflates on re-read, EXT-03 confirmed 3→7 curve inflation). L-02: non-first-section dedup never writes back global curves (inner writeback blocks are dead code — only call site passes False). N-I-20: emit ~{name} | {section_target} per LOG_DATA section, not hardcoded | CURVE. Checklist: dedup key = mnemonic+unit+format; dedup during first extension + warn on unit/format mismatch; parser-side registration dedup; per-section scope preserved through merge (P-03). In pylasdev-reborn: W-01(HIGH), L-02, N-I-20, EXT-03 — 4 CONFIRMED (co-located G8/G4b fix family).

## [pat-20260801161519-2c365e]
Category: pattern
Tags: python, performance, hot-path, memory, benchmark
Changed: 2026-08-01T16:15:19.059999

Per-value hot-path costs in data reading: hoist flag lookups, prefer math module over numpy for Python scalars, pre-allocate numeric columns, and resource guards must count intermediate phases. IT3-F-01: _desanitize_las_value re-runs full import machinery per value via thread-local __getattr__ (1.04-1.07 µs/value; 2.05x end-to-end read slowdown) — cache the flag once per read. IT3-F-02: np.isfinite/np.isnan/np.isinf on Python scalars are 15.3x slower than math.isfinite (semantically identical for Python floats) — swap per-value scalar checks to math module; leave array-vectorized numpy uses. IT3-F-03: _read_wrapped accumulates every value as Python float (~32B/value) then converts at end — 1.80x peak-RSS delta at 3M values; MAX_TOTAL_ELEMENTS counts final arrays only, so the intermediate list phase bypasses the OOM guard's intent — pre-allocate numeric columns or cap the intermediate phase. N-I-26: 9 unbounded str.split() sites in delimiter-detection bypass G-18 token-cap (102.8MB RSS growth on 0.58MB file, 177x amplification) — bound with maxsplit. Checklist: profile per-value loops; grep thread-local/import-machinery lookups in hot loops; math.* for Python scalar checks, np.* only for arrays; pre-allocate arrays; every str.split() bounded. In pylasdev-reborn: IT3-F-01, IT3-F-02, IT3-F-03, N-I-26 — 4 CONFIRMED (measured).

## [got-20260802053918-c67976]
Category: gotcha
Tags: python, encoding, detection, cyrillic, byte-analysis
Changed: 2026-08-02T05:39:18.889349

Encoding detection: character-based analysis on decoded samples is unreliable when the decoding itself maps bytes to ambiguous code points. In pylasdev-reborn _decode_best_quality(), a _has_cyrillic check inspecting candidates[0][1] (cp1251-decoded sample) for Cyrillic code points (U+0400-U+04FF) gave false positives for Western European files: cp1251 maps Western accented bytes (0xC0-0xFF) to Cyrillic code points. Fix: byte-frequency analysis on RAW bytes (before decoding). Count frequency of bytes corresponding to the top-10 most common Russian letters in cp1251 ({0xE0, 0xE2, 0xE5, 0xE8, 0xEB, 0xED, 0xEE, 0xF0, 0xF1, 0xF2}) — Russian text runs 25-57% frequency, typical Western text 2-5%. CORRECTION (2026-08-02, M-57 CONFIRMED): the 10% threshold is a heuristic, NOT a universal separator — accent-dense Western text false-positives: a realistic Spanish/Portuguese cp1252 file (ñ-dense) measures 15.7% > 10% and is misdecoded as cp1251 (mojibake, zero warnings). Do not treat the threshold as unambiguous; pair the byte-frequency detector with the other detectors (run-length, №-adjacency — see the Cyrillic detector-family pattern) and validate against accent-dense Western inputs.

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

## [pat-20260802053935-9b4b26]
Category: pattern
Tags: testing, regression, false-confidence, pre-fix, falsifiability
Changed: 2026-08-02T05:39:35.727803

Vacuous regression tests: a regression test that passes on PRE-FIX code provides zero regression value — it cannot catch the regression it claims to guard. The s9 test-quality pass found 5 vacuous tests guarding the highest-value fixes (H-04 ReDoS, H-01 scoping) with no effective protection. Five failure modes: (a) WRONG INPUT SHAPE — H-04 test used 100 lines x single '{'+550A (linear on pre-fix) instead of MANY unclosed braces on ONE line (the quadratic trigger, 22s pre-fix); (b) BRANCH ROUTING — H-01 test used pipe-qualified ~LOG_DATA + classic ~C, routing around the changed bare-~LOG_DATA branch; (c) WRONG DATA TYPE — M-32 test used Python lists (exact-equal, symmetric pre-fix) instead of ndarrays (tolerance-asymmetric pre-fix); (d) FALSE CONTRACT — M-76 space test had ZERO separators and asserted recombination the comma-gated code cannot perform; (e) WRONG ASSERTION TARGET — M-04 test asserted section data while the bug corrupted top-level logs, and used fill values == declared NULL so the buggy write path never executed. MECHANICAL GATE: run every new regression test against the PRE-FIX tree; if it passes, it is vacuous — fix input shape/branch/type/target until it fails on pre-fix and passes on post-fix. This is the operational test of the general rule in pat-20260721180622-4f55ad.

## [pat-20260802053941-5ce3d0]
Category: pattern
Tags: python, wrap-detection, las30, reader, contract, declared_wrap
Changed: 2026-08-02T05:39:41.198498

Wrap-detection two-path contract: wrap detection is implemented TWICE — LAS 1.2/2.0 path (data_reader._detect_actual_wrap) and LAS 3.0 path (_las30_data._detect_actual_wrap_las30) — and the two implementations drift, so a wrap fix must land on BOTH. This session: H-02 (2.0 majority-vote misclassifies when ~C declares MORE curves than ~A rows — every line short -> WRAPPED -> half the rows dropped, columns shifted), M-05 (WRAP=NO + genuinely wrapped data misparsed; 3.0 path has no content-based detection while 2.0 parses the same data correctly), M-07 (3.0 COMMA/TAB wrap decision made on FIRST data line only), M-38 (mixed-wrap first-line-full misdetected as non-wrapped, DEPT polluted — warnings fire but misdiagnose), F-04 (3.0 helper short-circuits return False on first-full-line with NO declared_wrap fall-through — dispatch gap), F-05 (H-02 uniform-short-row guard omitted in 3.0 helper -> WRAP=NO short-row files REJECTED with a factually wrong error, pre-fix parsed OK). Coherent contract: majority vote over >=3 lines, curve_count-aware, declared_wrap (WRAP=YES) fall-through in BOTH paths, content-based detection on 3.0 too, warnings that correctly attribute the failure mode (short-row vs misdetection). Fixes must be mirrored across both implementations — dispatching a wrap fix to one path alone is an asymmetric-fix regression. See pat-20260801161344-fc653b for the corroboration core.

## [pat-20260802053943-dd5df5]
Category: pattern
Tags: python, container-guard, reconcile, trim, grow, interaction
Changed: 2026-08-02T05:39:43.175885

Whole-container reconcile vs per-key guards: internal whole-container reconciliation (trimming or growing ALL curve arrays to a common length) must NOT trip per-assignment length guards — it needs an explicit bypass path. In pylasdev-reborn, the F-01 fix's F36 whole-container trim (las_file.logs[curve_name] = arr[:current_line]) tripped the M-43 per-key _check_value_length guard -> spurious crash (HIGH F-01); the FIX-CONV-2 G-04 grow uses dict.__setitem__ whole-container growth to preserve the equal-length invariant, mirroring _GuardedDict.trim_all's bypass (models.py:302-303). Design rule: when a guarded container enforces equal-length via per-key validation, whole-container resize operations (trim/grow/sync-all) bypass per-key guards; per-key mutation stays guarded. The invariant (equal-length) and the guard (per-key) must be separated. Checklist: when adding a per-key length guard, verify internal reconcile paths have a bypass; when adding a reconcile, use the bypass not per-key assignment; document the invariant-vs-guard separation; test both trim (overcount) and grow (undercount) directions.

## [pat-20260802053948-f668cb]
Category: pattern
Tags: python, numpy, int64, overflow, comparison, allclose, twos-complement
Changed: 2026-08-02T05:39:48.203459

int64 subtraction overflow in hand-rolled allclose: np.allclose promotes operands to float64 internally, but hand-rolled symmetric-allclose implementations that diff/abs in native dtype WRAP on int64 overflow (two's complement): [-2**63] vs [0] compares EQUAL (diff wraps to -2**63, abs(-2**63) is still -2**63 -> within rtol) when the values are plainly unequal. F-17 (compare.py _allclose_symmetric): int64 subtraction/abs overflow -> wrong True; -2^63 is a common int64 missing-sentinel and {I} int64 curves are produced by the library's own reader. The M-32 symmetric-allclose fix REINTRODUCED this class (np.allclose's implicit promotion was the only thing masking it before). Fix: promote BOTH operands to float64 (astype(np.float64)) BEFORE diff/abs, matching np.allclose semantics; MaskedArray operands must also be promoted (existing filled(np.nan) path). Checklist: in any comparison computing diff = a - b or abs(a - b) on integer dtypes, promote to float64 first; test with [-2**63] vs [0] and [2**63-1] vs [0]; verify symmetric argument swap gives the same answer; re-check when replacing np.allclose with a hand-rolled implementation.

## [pat-20260802053951-54b7bd]
Category: pattern
Tags: python, encoding, cyrillic, detection, mojibake, adjacency, false-positive
Changed: 2026-08-02T05:39:51.323343

Cyrillic detector family: coordination and adjacency-scoping. The Cyrillic-vs-Western decision in encoding.py uses THREE independent detectors (byte-frequency, run-length, №-adjacency) — each has a DISTINCT failure mode, and a detector added to fix one failure can introduce another: M-57 byte-frequency false-positives on accent-dense Western text (realistic ñ-dense cp1252 -> 15.7% > 10% threshold -> whole file misdecoded cp1251, mojibake, zero warnings); M-82 run-length false-positives on Western cp1252 with >=3 consecutive accented bytes (Ñáñez -> decoded as Cyrillic Сбсez); M-81 №-density gap (genuine cp1251 №-rich files fail when № density > ~4% -> misdecoded cp1252; chardet does not rescue, conf 0.03-0.20); F-18 №-confirmation must require 0xB9 ADJACENT to the Cyrillic run, not whole-file membership — a lone Western '¹' (also byte 0xB9 in cp1252) plus any accented run elsewhere flips the file (encoding.py now uses _NUMERO_ADJACENCY_WINDOW). Checklist: every added encoding detector must be tested against the OTHER detectors' known false-positive classes (accent-dense Western, №-dense Cyrillic, multi-consecutive-accent Western); a detector that only ADDS detection (no false-positive guard) will misroute realistic files; per-char byte signatures need positional (adjacency) constraints, not just presence. Extends pat-20260801161435-e13efb; corrects the byte-frequency threshold claim in got-20260802053918-c67976.

## [pat-20260802053958-275951]
Category: pattern
Tags: python, writer, parser, frozenset, order, column-swap, scoping
Changed: 2026-08-02T05:39:58.691584

Order-insensitive set comparison for order-sensitive output: using set/frozenset equality for scoping or identity decisions when column ORDER is semantically meaningful silently swaps columns. M-66/M-68 (_writer_las30.py): writer compares frozensets of curve names to decide whether a section's curves match the main block — {GR, DEPT} == {DEPT, GR}, so a section with the SAME curve-name set but DIFFERENT column order roundtrips with GR/DEPT data silently SWAPPED, zero corruption warnings. M-83: same dedup region, mnemonic-only key drops the second section's desc/api_code (W-01 compares unit/format only). The parser-side scope resolution (main_curve_end pipe branch, M-67/M-69) must mirror the writer's per-section scoping — H-01's fix did not cover the pipe branch, giving two failure directions (frozen-at-0 -> whole LOG_DATA discarded; unfrozen -> None -> phantom columns). Fix: when output ordering matters (per-section column order, scoping), compare order-sensitive structures (tuples, sequences) or emit explicit per-section definitions — never frozenset/set equality on ordered data. Checklist: grep frozenset/set comparisons in writer/parser scoping and dedup; verify ordering is preserved; test with same-set-different-order sections; dedup keys must include distinguishing attributes (unit/format/desc) not just the name.

## [got-20260802054001-11a495]
Category: gotcha
Tags: python, dev-reader, thousands, separator, locale, headerless, signed
Changed: 2026-08-02T05:40:01.554188

Thousands-separator recombination edge cases: _recombine_thousands_separators (dev_reader.py) has a family of silent-corruption cases beyond the basic 1,234.5: (a) MULTI-SEPARATOR values (>=1e6, 2+ separators) only PARTIALLY recombined — the len(values)==expected+1 gate merges the FIRST pair only, true value destroyed (M-76: 6-token no recombine at all; 5-token MD=1234.0, X=567.8); (b) SIGNED values fail the isdigit gate — '-1,234.5' skips its true pair and a later genuine adjacent pair is merged 600+500->600500, with the warning citing the WRONG pair (M-53); (c) HEADERLESS files never recombined — the gate requires non-empty names, and the M-52 fix's _expected_cols = len(names) if names else len(values)-1 made first headerless rows ALWAYS eligible, merging genuine 2-col headerless comma files into one column (F-07 regression); (d) the gate is DELIMITER-BLIND — semicolon locale-decimal values NaN with a misleading warning while detection says parseable (F-13). Fix: iterate consecutive pairs (not just the first), use a numeric-aware check (try float() not isdigit), gate recombination on unambiguous evidence (delimiter-aware AND not-headerless-first-row), and warn with the ACTUAL merged pair. Checklist: test multi-separator, signed, headerless-comma, and semicolon-locale inputs; a fix must be comma-gated AND header-aware; regression-test the pre-fix corruption shape.

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

## [got-20260802195912-698112]
Category: gotcha
Tags: wrap, data_reader, las30, curve-count, regression, parser
Changed: 2026-08-02T19:59:12.039379

F-07 wrap depth-line rule needs curve_count-aware gate: the 2-curve ambiguity (window[1]==1 after a full first row) is NOT unambiguous for curve_count==2 — the arm MUST be gated (curve_count >= 3 and window[1] == 1) or sum(1 for n in window[1:] if n == 1) >= 2, mirrored identically on BOTH wrap-detection paths (data_reader.py:641-644 AND _las30_data.py:364-367). PF-18 (HIGH regression): the s9 fix mirrored the las30 gate onto data_reader — the unconditional window[1]==1 arm misclassified a 2-curve LAS 1.2/2.0 WRAP=NO file with a short middle row (window [2,1,2]) as WRAPPED -> silent data corruption (genuine values discarded/misaligned). This SUPERSEDES the earlier refinement 'window[1]==1 OR >=2 one-value rows' which was curve_count-blind and regressed [2,1,2] nc=2. The >=2-one-value-rows arm alone catches genuine 2-curve wrapped files. Validated 12 shapes. Checklist: every wrap rule arm must be curve_count-aware; two-path parity is verified by comparing the gate expressions verbatim; regression-test the 2-curve WRAP=NO short-middle-row shape on both LAS 1.2 and LAS 2.0 plus string-padding.

## [got-20260802195917-35c9c7]
Category: gotcha
Tags: python, pickle, slots, containers, __setitem__, validation, models
Changed: 2026-08-02T19:59:17.208699

Pickle on guarded containers: EVERY __slots__ container subclass with __setitem__-time validation breaks unpickling (dumps OK / loads AttributeError), because default unpickling restores items through __setitem__ BEFORE the slot is set. _GuardedDict (models.py, __slots__ + unconditional _check_column_array_like at :369) was the last slotted container missing __reduce__; _GuardedList (:565-579), _DevColumns (:5232), _DevColumnOrder (:5409) already had it. PF-09 (MEDIUM regression): adding __setitem__-time validation to a slotted container breaks pickle unless __reduce__/__setstate__ is added in the SAME change — the guard fired before _container_name was restored. Fix: __reduce__ reconstructs via __init__(dict(self)) (re-validates + sets slot), __setstate__ restores the slot. Checklist: any slotted container whose __setitem__ touches an instance slot needs __reduce__/__setstate__; add a pickle roundtrip test that verifies guards still raise post-unpickle; test BOTH non-empty (items restored through __setitem__) and empty containers.

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

## [pat-20260802195949-d76275]
Category: pattern
Tags: python, writer, curves_order, column-swap, order-source, case-insensitive, regression
Changed: 2026-08-02T19:59:49.421096

Metadata emission must use the SAME live order source as data rows: when ~C (metadata) and data rows are emitted from DIFFERENT order sources (cached curves list vs live curves_order), a post-construction mutation silently swaps columns on write->read. PF-22 (MEDIUM, incomplete I2-13 fix): the legacy ~A path emitted ~C from the cached 'curves' list while data rows emitted from live 'curves_order' — post-construction curves_order mutation (I2-13) produced a file whose ~C order disagrees with its data columns, and re-read silently swaps columns, with only a suppressible models warning. PF-21 (MEDIUM, incomplete I2-22): same class on the LAS 3.0 path — case-insensitive lookup missing in writer fallback loops (curves_by_mnem :332, _section_mnems :435) -> false warnings, ~C reorder, duplicate emission. Fix: emit metadata from the same live order structure the data rows use; apply consistent key normalization (e.g. .upper()) in EVERY lookup path the emission uses, not just the primary one. Checklist: identify every order source feeding an emission; they must be the SAME object; apply identical normalization in all fallback/primary lookup loops; mutation-after-construction tests must cover the legacy path, not just LAS 3.0.

## [pat-20260802195954-b51168]
Category: pattern
Tags: python, writer, writeback, las30, copy-back, roundtrip, to_dict
Changed: 2026-08-02T19:59:54.100880

Copy-back/writeback must transfer ALL distinguishing fields, not just identity fields: when a fix propagates a curve/attribute to a top-level structure (writeback), copying only mnemonic+array_info but NOT the stripped description leaks state on roundtrip. PF-19 (MEDIUM, incomplete L30-01 fix): F2-07 writeback (_las30_data.py:891-905) copied mnemonic/array_info but not the stripped description -> to_dict leaked the {A:0} marker and the no-data_sections write path double-emitted '{A:0} {A:0}'. Fix: extend writeback to copy the stripped description (same source the section curve uses). Checklist: when writeback/copy-back copies a curve, enumerate ALL fields the roundtrip (parse->to_dict, write) depends on — identity (mnemonic/array_info) AND descriptive (description, unit, format) fields; test the no-data_sections write path AND parse->to_dict; the marker/format text must not leak into output twice.

## [got-20260802200007-ac0a13]
Category: gotcha
Tags: python, dev-reader, thousands-separator, performance, quadratic, dos
Changed: 2026-08-02T20:00:07.674857

Thousands-separator recombination must be linear single-pass: re-scanning or pair-restarting inside _recombine_thousands_separators makes it O(n^2) on long token rows. I2-18 (CONFIRMED MEDIUM): clean quadratic benchmark 0.10s -> 27.65s @ 16K tokens, ~18min @ 100K — a crafted file is a DoS. Fix: single linear pass with first-run exact-fit (only the FIRST pair that produces the expected column count is merged per the gate), per-pair warnings on subsequent merges. Complements the corruption-family gotcha (got-20260802054001-11a495): the fix must be BOTH correct (multi-separator/signed/headerless/delimiter-aware) AND linear. Checklist: benchmark recombination on >=16K-token rows after any change; avoid nested loops over the token list; the 'first pair only' gate must not degrade into per-pair restarts.

