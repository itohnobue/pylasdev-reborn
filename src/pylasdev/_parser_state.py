"""Mutable parser state extracted from LASParser.

Contains all mutable state that was previously reset per-file in the
``LASParser._reset()`` method.  Separating state from parsing logic
enables cleaner testing (state can be constructed independently) and
makes the `_SectionTransitionHandler`'s dependency on parser state
explicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import LASFile

logger = logging.getLogger(__name__)


@dataclass
class _ParserState:
    """Mutable parser state extracted from LASParser.

    Contains all mutable state that was reset per-file in the original
    ``LASParser._reset()`` method.  Separating state from parsing logic
    enables cleaner testing (state can be constructed independently),
    removes the 500-line ``__init__``/``_reset`` block from ``LASParser``,
    and makes the `_SectionTransitionHandler`'s dependency on parser
    state explicit.

    Instance Attributes:
        current_section: Current section type code (``"V"``, ``"W"``,
            ``"C"``, ``"P"``, ``"O"``, ``"A"`` or ``None``).
        current_section_name: Human-readable name for the current section.
        current_data_section_type: LAS 3.0 data section type string
            (e.g. ``"LOG_DATA"``).
        current_data_section_idx: Zero-based data section counter.
        current_definition_name: Active ``_Definition`` section name
            (e.g. ``"CORE_DEFINITION"``) or ``None``.
        data_line_count: Pre-scanned data line count from ``_pre_scan``.
        ascii_data_lines: Collected ASCII data lines for the current section.
        cumulative_elements: Running total of elements across data sections.
        cumulative_data_lines: Running total of data lines across sections.
        cumulative_data_lines_warned: Warning-issued-once flag for
            cumulative line counts.
        version_found: ``True`` after ``VERS`` line in ``~V`` section.
        other_lines: Accumulated ``~O`` (other) section text.
        section_curve_start_idx: Start index into ``las_file.curves``
            for the current section's curve scope.
        section_curve_end_idx: End index (exclusive) for current section
            curve scope, or ``None``.
        main_curve_end: End of the main (non-``_Definition``) curve block.
        definition_curve_ranges: Per-definition-name → (start_idx, end_idx)
            mapping.
        las30_sections_seen: ``True`` if any LAS 3.0 typed data section
            was encountered.
        deferred_well_entries: Buffered raw well entries parsed before
            ``~V`` was known.
        deferred_ascii_data_lines: Buffered raw data lines parsed before
            ``~V`` was known.
        section_sequence: Ordered list of section header labels.
        section_type_sequence: Ordered list of section type codes
            (parallel to ``section_sequence``).
    """

    # --- Current section tracking ------------------------------------------
    current_section: str | None = None
    current_section_name: str = ""
    current_data_section_type: str = "LOG_DATA"
    current_data_section_idx: int = 0
    current_definition_name: str | None = None

    # --- Pre-scan ----------------------------------------------------------
    data_line_count: int = 0

    # --- ASCII data accumulation -------------------------------------------
    ascii_data_lines: list[str] = field(default_factory=list)

    # --- Cumulative counters -----------------------------------------------
    cumulative_elements: int = 0
    cumulative_data_lines: int = 0
    cumulative_data_lines_warned: bool = False

    # --- Version tracking --------------------------------------------------
    version_found: bool = False

    # --- Other section accumulation ----------------------------------------
    other_lines: list[str] = field(default_factory=list)

    # --- Curve scope state -------------------------------------------------
    section_curve_start_idx: int = 0
    section_curve_end_idx: int | None = None
    main_curve_end: int = -1
    definition_curve_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    # --- LAS 3.0 state -----------------------------------------------------
    las30_sections_seen: bool = False

    # --- Deferred parsing state --------------------------------------------
    deferred_well_entries: list[dict[str, str]] = field(default_factory=list)
    deferred_ascii_data_lines: list[
        tuple[str, str, int, str, int, int | None]
    ] = field(default_factory=list)

    # --- Section tracking --------------------------------------------------
    section_sequence: list[str] = field(default_factory=list)
    section_type_sequence: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """Reset all state for a new file.

        Sets every field back to its default value.  List and dict
        fields are cleared in-place to avoid identity changes that
        could break held references.
        """
        self.current_section = None
        self.current_section_name = ""
        self.current_data_section_type = "LOG_DATA"
        self.current_data_section_idx = 0
        self.current_definition_name = None
        self.data_line_count = 0
        self.ascii_data_lines.clear()
        self.cumulative_elements = 0
        self.cumulative_data_lines = 0
        self.cumulative_data_lines_warned = False
        self.version_found = False
        self.other_lines.clear()
        self.section_curve_start_idx = 0
        self.section_curve_end_idx = None
        self.main_curve_end = -1
        self.definition_curve_ranges.clear()
        self.las30_sections_seen = False
        self.deferred_well_entries.clear()
        self.deferred_ascii_data_lines.clear()
        self.section_sequence.clear()
        self.section_type_sequence.clear()

    def validate(self, las_file: LASFile) -> list[str]:
        """Cross-validate state consistency.

        Checks invariants that span multiple fields and cannot be
        enforced at individual assignment points.  Called once after
        parsing completes (from ``LASParser.parse()``) to catch
        inconsistencies before returning the ``LASFile``.

        Args:
            las_file: The ``LASFile`` being built (needed for invariants
                that cross-reference parser state against the output).

        Returns:
            List of issue strings (empty list if state is consistent).
        """
        issues: list[str] = []
        self._check_las30_section_consistency(issues, las_file)
        self._check_data_section_idx_consistency(issues)
        self._check_deferred_state_consistency(issues)
        self._check_section_sequence_consistency(issues)
        self._check_cumulative_counters_consistency(issues)
        return issues

    # ------------------------------------------------------------------
    # Invariant checks
    # ------------------------------------------------------------------

    def _check_las30_section_consistency(
        self, issues: list[str], las_file: LASFile
    ) -> None:
        """Invariant 1+3: LAS 3.0 sections vs VERS + data sections vs is_las30."""
        # Invariant 1: LAS 3.0 sections seen but version is not LAS 3.0.
        if self.las30_sections_seen and not las_file.version.is_las30:
            issues.append(
                f"LAS 3.0 structured data sections found but VERS is "
                f"'{las_file.version.vers}' (not a 3.x version) — "
                f"LAS 3.0 data handling is DISABLED."
            )
        # Invariant 3: Data sections present but is_las30 is False.
        if las_file.data_sections and not las_file.version.is_las30:
            issues.append(
                f"LASFile has {len(las_file.data_sections)} data section(s) "
                f"but is_las30 is False — data sections are only valid for "
                f"LAS 3.0."
            )

    def _check_data_section_idx_consistency(self, issues: list[str]) -> None:
        """Invariant 2: Data section index vs dangling ASCII data lines."""
        if self.current_data_section_idx > 0 and self.ascii_data_lines:
            issues.append(
                f"current_data_section_idx is {self.current_data_section_idx} "
                f"but ascii_data_lines still has {len(self.ascii_data_lines)} "
                f"un-flushed lines — dangling data would produce corrupt output."
            )

    def _check_deferred_state_consistency(self, issues: list[str]) -> None:
        """Invariant 5: Deferred buffers must be empty after parse completes."""
        if self.deferred_well_entries:
            issues.append(
                f"deferred_well_entries still contains "
                f"{len(self.deferred_well_entries)} entries after parse "
                f"completed — _replay_deferred_well() may not have run."
            )
        if self.deferred_ascii_data_lines:
            issues.append(
                f"deferred_ascii_data_lines still contains "
                f"{len(self.deferred_ascii_data_lines)} lines after parse "
                f"completed — _replay_deferred_well() may not have run."
            )

    def _check_section_sequence_consistency(self, issues: list[str]) -> None:
        """Invariant 7: Section sequence and type sequence must stay synchronized."""
        if len(self.section_sequence) != len(self.section_type_sequence):
            issues.append(
                f"section_sequence ({len(self.section_sequence)} entries) and "
                f"section_type_sequence ({len(self.section_type_sequence)} "
                f"entries) are out of sync — the two parallel lists must "
                f"have equal length."
            )

    def _check_cumulative_counters_consistency(self, issues: list[str]) -> None:
        """Invariant 8: Cumulative data line counter vs warning flag."""
        if (
            self.cumulative_data_lines > 0
            and self.cumulative_data_lines_warned
            and self.current_data_section_idx <= 1
        ):
            issues.append(
                f"Cumulative data line warning raised with only "
                f"{self.current_data_section_idx + 1} data section(s) — "
                f"the warning threshold requires multiple sections to be "
                f"reachable; this may indicate a corrupted counter."
            )
