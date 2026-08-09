"""Section-transition lifecycle handler for LASParser.

Extracted from parser.py:LASParser._parse_line.  Handles the three-phase
section-transition lifecycle:

    1. capture_current_state()  → _CapturedState
    2. process_previous_section(captured, new_section) → cumulative_elements
    3. enter_new_section(section_type, section_label, section_word, section_rest)

TYPE-LEVEL ORDERING: ``process_previous_section`` REQUIRES a
``_CapturedState`` parameter — you cannot call it without first calling
``capture_current_state``.

The handler owns the transition logic.  The parser owns classification
and line dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LASFile
    from .parser import LASParser

logger = logging.getLogger(__name__)


def _ascii_section_display_name(section_name: str | None) -> str:
    """Shared M-22 naming rule for standard ~A/~ASCII data sections.

    A standard ASCII data section header without an explicit name — a
    bare ``~A``/``~ASCII``, or ``~ASCII | <pipe>`` where the pipe is a
    curve/definition association rather than a name — is left unnamed so
    the LAS 3.0 DataSection machinery auto-names it ``Section_N``
    (``_las30_data.py:861``).  An explicit name (e.g. ``~A FirstSection``)
    is preserved exactly — EXCEPT an explicit name that itself IS a bare
    keyword (``A``/``ASCII``, case-sensitive): ``~A A``/``~A ASCII`` is
    treated as a bare keyword and auto-named ``Section_N``, matching the
    deferred path's ``_DATA_SECTION_WORDS`` blanking (parser.py:3257).
    All other explicit names are preserved exactly.

    PARS-C-PROD (M-22 violation): the direct path previously fell back
    to ``section_word`` for bare ``~A``/``~ASCII`` headers, producing two
    DataSections both named ``'A'``/``'ASCII'`` — a spurious "duplicate
    data section name" on ``validate(complete=True)`` — while the
    deferred pre-~V path (parser.py:3250-3257) auto-named them
    ``Section_0``/``Section_1``.  Both paths must produce the SAME names
    for the same file shape (M-22 order-invariance contract).  The
    deferred path's broader bare-keyword blanking covers this family
    (``_DATA_SECTION_WORDS`` membership); this helper is the direct
    path's implementation of the same rule for the standard family.
    """
    if not section_name:
        return ""
    stripped = section_name.strip()
    # A name that IS a bare keyword ("~A ASCII") is not a user-provided
    # name — the deferred path blanks it too (case-sensitive membership
    # in _DATA_SECTION_WORDS), so keep the two paths in lockstep.
    if stripped in {"A", "ASCII"}:
        return ""
    return stripped


@dataclass(frozen=True)
class _CapturedState:
    """Snapshot of parser state at the point a new section header is detected.

    Frozen so it cannot be mutated after capture — the transition handler
    processes the captured snapshot, not live parser state.
    """

    # The section we are LEAVING (None if first section)
    previous_section: str | None

    # ASCII data accumulated in the previous section
    ascii_data_lines: list[str]

    # Curve range of the previous section (for per-section scoping in LAS 3.0)
    curve_start_idx: int
    curve_end_idx: int | None

    # Metadata of the previous section
    section_name: str
    data_section_type: str

    # Whether ~V has been parsed (controls deferred-well replay)
    version_found: bool

    # The LASFile being built (needed for AsciiDataContext)
    las_file: LASFile

    # Previous definition name — for saving C section curve ranges
    previous_definition_name: str | None

    # PF-01: Pipe target of the PREVIOUS (leaving) data section, captured
    # BEFORE classification resets ``_current_pipe_target`` to the NEW
    # section's value (parser.py:1468).  The A→A consecutive-data path must
    # restore this so ``_flush_ascii_data``'s pre-~V deferral records the
    # FIRST section's forward "| X_Definition" pipe (PARS-06) instead of the
    # second section's (bare → None) target.
    current_pipe_target: str | None = None


class _SectionTransitionHandler:
    """Handles the three-phase section-transition lifecycle.

    TYPE-LEVEL ORDERING (enforced by the API):
        1. capture_current_state() → _CapturedState
        2. process_previous_section(captured, new_section) → int
        3. enter_new_section(section_type, ...) → None

    You CANNOT call ``process_previous_section`` without first calling
    ``capture_current_state`` — the method signature requires the captured
    object.

    The handler holds a reference to the parser to read/write parser state.
    It owns the transition logic; the parser owns classification and line
    dispatch.
    """

    def __init__(self, parser: LASParser) -> None:
        self._parser = parser

    # ------------------------------------------------------------------
    # Phase 1 — CAPTURE: snapshot parser state BEFORE classification
    # ------------------------------------------------------------------

    def capture_current_state(self) -> _CapturedState:
        """Snapshot the current section's state BEFORE classification runs.

        Must be called at the TOP of the new-section-detection block,
        before classification determines ``new_section`` and overwrites
        ``_current_definition_name``, ``_current_data_section_type``, etc.

        Also saves the current C section's curve range to
        ``_definition_curve_ranges`` so pipe-target lookups during
        classification can find the freshly-saved entry (H-03/H-01).

        Returns:
            Frozen _CapturedState with all data needed to finalize the
            previous section.
        """
        p = self._parser

        # Save C curve range to _definition_curve_ranges BEFORE classification
        # runs.  This is needed so pipe-target lookups in the data-section
        # classification block can find the entry (H-03/H-01).
        if p._state.current_section == "C":
            start = p._state.section_curve_start_idx
            end = (
                p._state.section_curve_end_idx
                if p._state.section_curve_end_idx is not None
                else len(p.las_file.curves)
            )
            if p._state.current_definition_name is not None:
                p._state.definition_curve_ranges[p._state.current_definition_name] = (start, end)
            else:
                # H-01: Non-_Definition ~C section — save under sentinel.
                # PARS-09: A "| CURVE" pipe must see the UNION of ALL repeated
                # plain ~C blocks (``~C(DEPT,GR) ~C(RHOB) ~A|CURVE`` previously
                # resolved to a truncated (2,3) scope and silently discarded
                # DEPT/GR).  __MAIN_ALL__ accumulates the union; __MAIN__ keeps
                # last-writer-wins for the BARE-~A fallback (F-S9-02: a bare
                # data section scopes to the MOST RECENT plain ~C block).
                # M-14: the union is a min/max over plain-~C ranges and is
                # therefore only valid while those ranges are CONTIGUOUS.
                # An interleaved _Definition/_Data block sits INSIDE the
                # min..max span (curve indices are global file order), so
                # merging across the gap would leak the definition curves
                # into the "| CURVE" scope (silent data misattribution: the
                # genuine log column nulled, its values stored under the
                # _Definition name).  Only merge when the new plain-~C range
                # is contiguous with the accumulated union; a plain-~C block
                # after a _Definition is NOT part of the main scope (its
                # columns degrade to a loud extra-column warning instead of
                # silent misattribution).
                _prev_all = p._state.definition_curve_ranges.get("__MAIN_ALL__")
                if _prev_all is not None and start <= _prev_all[1]:
                    _all_start = min(_prev_all[0], start)
                    _all_end = max(_prev_all[1], end)
                elif _prev_all is not None:
                    _all_start, _all_end = _prev_all
                else:
                    _all_start, _all_end = start, end
                p._state.definition_curve_ranges["__MAIN_ALL__"] = (_all_start, _all_end)
                p._state.definition_curve_ranges["__MAIN__"] = (start, end)

        return _CapturedState(
            previous_section=p._state.current_section,
            ascii_data_lines=list(p._state.ascii_data_lines),  # shallow copy
            curve_start_idx=p._state.section_curve_start_idx,
            curve_end_idx=p._state.section_curve_end_idx,
            section_name=p._state.current_section_name,
            data_section_type=p._state.current_data_section_type,
            version_found=p._state.version_found,
            las_file=p.las_file,
            previous_definition_name=p._state.current_definition_name,
            # PF-01: Capture the pipe target at snapshot time — the
            # classification block that runs AFTER capture resets it to the
            # NEW section's target (parser.py:1468).  getattr guard: tests
            # that monkeypatch _reset() (R-005 format-vs-placement path) do
            # not initialize this attribute; a missing target is equivalent
            # to "no pipe on the previous section" (mirrors the
            # _section_pipe_targets guard in parser.py).
            current_pipe_target=getattr(p, "_current_pipe_target", None),
        )

    # ------------------------------------------------------------------
    # Phase 2 — PROCESS: finalize the previous section
    # ------------------------------------------------------------------

    def process_previous_section(self, captured: _CapturedState, new_section: str | None) -> int:
        """Process the previous section's accumulated data and finalize its state.

        REQUIRES: ``captured`` from a prior ``capture_current_state()`` call.
        The captured state is immutable — the handler works with the snapshot,
        not live parser state (except for the A→A swap which needs the NEW
        section's state already set by classification).

        Args:
            captured: Snapshot from ``capture_current_state()``.
            new_section: The section type determined by classification
                (``"V"``, ``"W"``, ``"C"``, ``"P"``, ``"O"``, ``"A"``,
                or ``None`` for unknown sections).

        Returns:
            Updated ``cumulative_elements`` count (to write back to the parser).
        """
        p = self._parser

        if new_section is not None:
            prev_sec = captured.previous_section

            if prev_sec == "A" and new_section != "A":
                # A→non-A: process the previous data section.
                self._process_ascii_section(captured)

            elif new_section == "A" and prev_sec == "A":
                # A→A (consecutive data sections): swap back to the
                # previous section's state, process, then restore the
                # new section's state.
                self._process_consecutive_data(captured)

            # Save the PREVIOUS C section's curve range to
            # _definition_curve_ranges.  Uses captured values
            # (snapshot before classification) so pipe overwrites
            # don't corrupt the range (F-01/G-02/H-03).
            if prev_sec == "C":
                self._save_c_curve_range(captured)

        return p._state.cumulative_elements

    # ------------------------------------------------------------------
    # Phase 3 — ENTER: set up parser state for the new section
    # ------------------------------------------------------------------

    def enter_new_section(
        self,
        section_type: str,
        section_label: str,
        section_word: str,
        section_name: str,
    ) -> None:
        """Enter the new section — reset parser state for the incoming section.

        Args:
            section_type: Normalized type code (``"V"``, ``"W"``, ``"C"``,
                ``"P"``, ``"O"``, ``"A"``).
            section_label: Full label for ``_section_sequence``
                (e.g. ``"CORE_DEFINITION:Core Definition"``).
            section_word: Raw section-word from the header
                (e.g. ``"CORE_DEFINITION"``).
            section_name: Computed section name from classification
                (may include section_word + section_rest).
        """
        p = self._parser

        # M-25: lazy import of the parser's keyword tables (same pattern as
        # MAX_SECTION_SEQUENCE below — _section_transition is imported by
        # parser at module load, so a module-level import would be circular).
        from .parser import _DATA_SECTION_WORDS, _is_indexed_data_section

        # F1: Track per-section curve boundaries for LAS 3.0.
        # When entering ~C (including _Definition sections), mark the
        # current curve list position for per-section curve scoping.
        if section_type == "C":
            p._state.section_curve_start_idx = len(p.las_file.curves)
            p._state.section_curve_end_idx = None

        p._state.current_section = section_type

        # F-M27: For parameter sections, _current_section_name must
        # preserve the section_word (e.g., CORE_PARAMETERS) for type
        # derivation in _parse_parameter_entry.  section_name may be
        # annotation text, not the type identifier.
        if section_type == "P" and (
            section_word.endswith("_PARAMETER") or section_word.endswith("_PARAMETERS")
        ):
            p._state.current_section_name = section_word
        elif section_type == "A" and section_word in {"A", "ASCII"}:
            # PARS-C-PROD / M-22: a standard ~A/~ASCII data section
            # header without an explicit name (bare "~A"/"~ASCII", or
            # "~ASCII | <pipe>") is left unnamed so the LAS 3.0 machinery
            # auto-names it Section_N — identical to the deferred pre-~V
            # path (parser.py:3250-3257).  Falling back to section_word
            # produced duplicate 'A'/'ASCII' names for two bare ~A
            # sections (spurious validate duplicate-name failure).  The
            # shared rule lives in _ascii_section_display_name.
            p._state.current_section_name = _ascii_section_display_name(section_name)
        else:
            _candidate = section_name.strip() if section_name else section_word
            # M-25 (PARS-C-PROD adjacent): non-A/ASCII bare keywords
            # (~CORE, ~LOG, ~TOPS_DATA, ~CORE[1], ...) previously fell
            # back to the keyword name on the DIRECT path, so two bare
            # ~CORE sections were both named 'CORE' — a spurious
            # "duplicate data section name" validate(complete=True)
            # failure — while the deferred pre-~V path (parser.py:3250-
            # 3257) blanked the whole _DATA_SECTION_WORDS family to
            # Section_N.  Blank to Section_N ONLY when the keyword name
            # would DUPLICATE an existing data section; single-occurrence
            # keyword names stay (pinned: DRILLING / CORE[1] /
            # PERFORATIONS / PERFORATIONS_DATA in test_reader.py, R7F-04
            # in test_parser.py).  Do NOT blanket-blank — that breaks the
            # pinned single-occurrence names.
            if (
                section_type == "A"
                and _candidate
                and (
                    _candidate in _DATA_SECTION_WORDS
                    or _is_indexed_data_section(_candidate)
                )
                and any(ds.name == _candidate for ds in p.las_file.data_sections)
            ):
                p._state.current_section_name = ""
            else:
                p._state.current_section_name = _candidate

        # F-34: Track section sequence for cross-section validation.
        # Section label uses section_word (e.g. "CURVE", "VERSION")
        # instead of just new_section (single letter "C", "V")
        # so that ~C and ~CURVE produce distinct labels (F-I2-M11).
        from .parser import MAX_SECTION_SEQUENCE

        if len(p._state.section_sequence) >= MAX_SECTION_SEQUENCE:
            from .exceptions import LASParseError

            raise LASParseError(
                f"Section sequence length ({len(p._state.section_sequence) + 1}) "
                f"exceeds maximum allowed ({MAX_SECTION_SEQUENCE}). "
                f"The file may be malformed or corrupt."
            )
        p._state.section_sequence.append(section_label)

        # F-048/F-103: Track semantic section type for duplicate detection.
        p._state.section_type_sequence.append(section_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_ascii_section(self, captured: _CapturedState) -> None:
        """Process accumulated ASCII data for a data section transition.

        Args:
            captured: The snapshot from ``capture_current_state()``.
        """
        p = self._parser

        # F-30: Delegate to the shared flush helper on LASParser to
        # eliminate duplicated AsciiDataContext construction and finally-
        # cleanup logic between this method and the unknown-section handler.
        p._flush_ascii_data(
            data_lines=captured.ascii_data_lines,
            section_curve_start_idx=captured.curve_start_idx,
            section_curve_end_idx=captured.curve_end_idx,
            current_section_name=captured.section_name,
            current_data_section_type=captured.data_section_type,
            current_data_section_idx=p._state.current_data_section_idx,
            cumulative_elements=p._state.cumulative_elements,
            version_found=captured.version_found,
            las_file=captured.las_file,
        )

    def _process_consecutive_data(self, captured: _CapturedState) -> None:
        """Handle consecutive data section (A→A) transition.

        The classification block has already overwritten
        ``_section_curve_start_idx``, ``_section_curve_end_idx``, and
        ``_current_data_section_type`` with the NEW section's values.
        This method saves the new values, restores the old values
        (from ``captured``), processes the previous section's data,
        and restores the new values.
        """
        p = self._parser

        # Save new (incoming) section's curve indices.
        _new_curve_start = p._state.section_curve_start_idx
        _new_curve_end = p._state.section_curve_end_idx

        # Restore old (previous) section's curve indices.
        p._state.section_curve_start_idx = captured.curve_start_idx
        p._state.section_curve_end_idx = captured.curve_end_idx

        # PF-01: Classification has already overwritten _current_pipe_target
        # with the NEW section's target (parser.py:1468).  Save it, restore
        # the PREVIOUS section's captured target so _flush_ascii_data's
        # pre-~V deferral records the first section's forward "| X_Definition"
        # pipe (PARS-06), then restore the new target on exit.
        _new_pipe_target = p._current_pipe_target
        p._current_pipe_target = captured.current_pipe_target

        try:
            # Swap data section type: save new, restore old.
            _new_type = p._state.current_data_section_type
            p._state.current_data_section_type = captured.data_section_type

            self._process_ascii_section(captured)

            # Restore new section's type.
            p._state.current_data_section_type = _new_type
        finally:
            # Restore new section's pipe target and curve indices and reset
            # accumulators.
            p._current_pipe_target = _new_pipe_target
            p._state.section_curve_start_idx = _new_curve_start
            p._state.section_curve_end_idx = _new_curve_end

    def _save_c_curve_range(self, captured: _CapturedState) -> None:
        """Save the previous C section's curve range.

        Uses captured values (snapshot before classification) so that
        pipe-handling overwrites of ``_section_curve_start_idx`` etc.
        don't corrupt the range (F-01).
        """
        p = self._parser

        start = captured.curve_start_idx
        end = (
            captured.curve_end_idx if captured.curve_end_idx is not None else len(p.las_file.curves)
        )

        if captured.previous_definition_name is not None:
            # G-02/H-03: Save under the definition name captured
            # before classification overwrites _current_definition_name.
            p._state.definition_curve_ranges[captured.previous_definition_name] = (start, end)
        else:
            # H-01: Non-_Definition ~C section — save under sentinel.
            # PARS-09: keep __MAIN__ last-writer-wins for the bare-~A
            # fallback, but accumulate __MAIN_ALL__ (union) for "| CURVE"
            # pipe resolution (see capture_current_state).
            # M-14: only merge CONTIGUOUS plain-~C ranges (see
            # capture_current_state) — never span an interleaved
            # _Definition block's curves.
            _prev_all = p._state.definition_curve_ranges.get("__MAIN_ALL__")
            if _prev_all is not None and start <= _prev_all[1]:
                _all_start = min(_prev_all[0], start)
                _all_end = max(_prev_all[1], end)
            elif _prev_all is not None:
                _all_start, _all_end = _prev_all
            else:
                _all_start, _all_end = start, end
            p._state.definition_curve_ranges["__MAIN_ALL__"] = (_all_start, _all_end)
            p._state.definition_curve_ranges["__MAIN__"] = (start, end)
