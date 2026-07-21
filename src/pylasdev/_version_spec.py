"""Centralized LAS version specification rules.

Consolidates scattered version-check patterns (``startswith("1.")``,
``startswith("2.")``, ``is_las30``, etc.) into a single frozen dataclass
so that LAS version-specific rules have one authoritative source.

All version-dependent logic across parser.py, writer.py, and models.py
delegates to the properties and methods on this class.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _LASVersionSpec:
    """Single source of truth for LAS version-specific rules.

    Constructed from a normalized version string (e.g. ``"1.2"``, ``"2.0"``,
    ``"3.0"``).  All boolean properties are derived from *vers*, and semantic
    rules (line-length limits, delimiter restrictions, section structure)
    are expressed as named properties instead of inline prefix checks.
    """

    vers: str
    """Normalized version string (e.g. ``"1.2"``, ``"2.0"``, ``"3.0"``)."""

    # --- identity checks ---------------------------------------------------

    @property
    def is_las12(self) -> bool:
        """True when *vers* starts with ``"1."`` (LAS 1.2 or 1.x variant)."""
        return self.vers.startswith("1.")

    @property
    def is_las20(self) -> bool:
        """True when *vers* starts with ``"2."`` (LAS 2.0)."""
        return self.vers.startswith("2.")

    @property
    def is_las30(self) -> bool:
        """True when *vers* starts with ``"3"`` (LAS 3.0 or 3.x variant).

        Uses ``"3"`` prefix (no dot) to match the existing ``VersionSection.is_las30``
        behaviour, which accepts ``"3.0"``, ``"3.1"``, ``"3.1beta"``, etc.
        """
        return self.vers.startswith("3")

    @property
    def is_las12_or_later(self) -> bool:
        """True for any recognised LAS version string (``"1."``, ``"2."``, or ``"3."``).

        Replaces ``self.las_file.version.vers.startswith(("1.", "2.", "3."))``
        used for mandatory well-field validation in the parser.
        """
        return self.is_las12 or self.is_las20 or self.is_las30

    # --- delimiter rules ---------------------------------------------------

    @property
    def allows_non_space_dlm(self) -> bool:
        """LAS 1.2 forces SPACE delimiter; 2.0+ allow TAB/COMMA."""
        return not self.is_las12

    @property
    def emits_dlm_line(self) -> bool:
        """DLM line in ~V section is only emitted for LAS 2.0+ (not LAS 1.2)."""
        return not self.is_las12

    # --- wrap / line-length rules ------------------------------------------

    @property
    def supports_wrap(self) -> bool:
        """LAS 3.0 forces WRAP=NO; 1.2 and 2.0 accept YES/NO."""
        return not self.is_las30

    def line_length_limit_for_wrap(self, wrap: str | None) -> int | None:
        """Return the line-length limit for the given WRAP mode, or None.

        LAS 1.2: always 256 chars.
        LAS 2.0: 256 chars when WRAP=NO; unlimited for WRAP=YES.
        LAS 3.0: no explicit limit (return None).

        Replaces the two-phase ``check_line_limit`` computation in
        ``writer._write_ascii_sections``.
        """
        if self.is_las12:
            return 256
        if self.is_las20 and (wrap or "NO").upper() == "NO":
            return 256
        return None

    # --- well-section rules ------------------------------------------------

    @property
    def well_format_swap_value_desc(self) -> bool:
        """LAS 1.2 CWLS well format: non-mandatory fields use DESC:VALUE ordering.

        When True, the parser's ``_store_well_entry`` applies the CWLS/lasio
        swap logic for non-mandatory well fields.
        """
        return self.is_las12

    @property
    def mandatory_well_fields(self) -> tuple[str, ...]:
        """Mandatory well-section field mnemonics for the current version.

        LAS 1.2, 2.0, and 3.0 all require STRT, STOP, STEP, NULL.
        LAS 1.2 additionally requires WELL, LOC, SRVC, UWI (soft-required).
        """
        if self.is_las12:
            return ("STRT", "STOP", "STEP", "NULL", "WELL", "LOC", "SRVC", "UWI")
        return ("STRT", "STOP", "STEP", "NULL")

    # --- section structure rules -------------------------------------------

    @property
    def has_structured_sections(self) -> bool:
        """LAS 3.0 has typed data/parameter sections (~Core, ~Log_Definition, etc.)."""
        return self.is_las30

    @property
    def supports_data_sections(self) -> bool:
        """Multi-section typed data only valid in LAS 3.0."""
        return self.is_las30

    @property
    def supports_other_section(self) -> bool:
        """~Other is deprecated and not allowed in LAS 3.0."""
        return not self.is_las30

    @property
    def supports_data_format(self) -> bool:
        """LAS 3.0 supports {F}, {E}, {A:0} data-format specifiers."""
        return self.is_las30

    @property
    def supports_zone_notation(self) -> bool:
        """LAS 3.0 supports | ZONE[idx] pipe notation."""
        return self.is_las30

    # --- index curve rules -------------------------------------------------

    @property
    def requires_index_curve(self) -> bool:
        """LAS 2.0 spec requires first curve to be DEPT/DEPTH/TIME/INDEX."""
        return self.is_las20

    # --- section duplication rules -----------------------------------------

    @property
    def duplicate_allowed_sections(self) -> frozenset[str]:
        """Section types that MAY appear more than once.

        LAS 1.2/2.0: only V, W, O are single-occurrence; C, P, A may repeat.
        LAS 3.0: V, W, O are single-occurrence; C, P, A may repeat.
        (In LAS 3.0 the parser already allows multiple ~C, ~P, ~A via typed sections.)
        """
        if self.is_las30:
            return frozenset({"C", "P", "A"})
        return frozenset()

    @property
    def single_section_types(self) -> frozenset[str]:
        """Section types that MUST appear at most once.

        LAS 1.2/2.0: V, W, O, C, P, A.
        LAS 3.0: V, W, O (C/P/A may repeat per typed group).
        """
        if self.is_las30:
            return frozenset({"V", "W", "O"})
        return frozenset({"V", "W", "O", "C", "P", "A"})
