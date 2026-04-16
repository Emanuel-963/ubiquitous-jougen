"""Reusable widget helpers and pure-logic utilities for the IonFlow GUI.

This module contains UI-agnostic helpers that can be tested headlessly,
plus lightweight base classes for chart tabs and table management.
The actual tkinter / customtkinter widget construction stays in the
individual tab modules (Day 14+).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  LogRedirector — write-protocol adapter for queue-based logging
# ═══════════════════════════════════════════════════════════════════════

class LogRedirector:
    """A file-like writer that pushes non-empty messages to a callback.

    This replaces the ``QueueWriter`` class by being queue-agnostic:
    the caller supplies *any* callback (``queue.put``, ``list.append``,
    ``logger.info``, …).
    """

    def __init__(self, callback: Callable[[str], Any]) -> None:
        self._callback = callback

    def write(self, msg: str) -> None:
        """Forward *msg* to the callback if it contains non-whitespace content."""
        if msg.strip():
            self._callback(msg)

    def flush(self) -> None:  # noqa: D401
        """No-op that satisfies the writable file protocol; returns ``None`` immediately."""
        return None


# ═══════════════════════════════════════════════════════════════════════
#  ChartExporter — save / copy a matplotlib figure or a static plot
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExportResult:
    """Outcome of a chart export operation."""

    success: bool
    path: str = ""
    error: str = ""


class ChartExporter:
    """Handles saving matplotlib figures and copying static plot files."""

    @staticmethod
    def save_figure(
        fig: Any,
        dest_path: str,
        *,
        dpi: int = 150,
        bbox_inches: str = "tight",
    ) -> ExportResult:
        """Save a matplotlib *Figure* to *dest_path* as PNG, SVG, or PDF.

        Creates intermediate directories if needed and returns an
        :class:`ExportResult` indicating success or failure.
        """
        try:
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            fig.savefig(dest_path, dpi=dpi, bbox_inches=bbox_inches)
            return ExportResult(success=True, path=dest_path)
        except Exception as exc:
            return ExportResult(success=False, error=str(exc))

    @staticmethod
    def copy_image(src_path: str, dest_dir: str) -> ExportResult:
        """Copy an existing image file at *src_path* into *dest_dir*.

        Returns an :class:`ExportResult` with the destination path on
        success, or an error message if the source file is missing.
        """
        if not os.path.isfile(src_path):
            return ExportResult(
                success=False, error=f"Arquivo não encontrado: {src_path}"
            )
        try:
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest)
            return ExportResult(success=True, path=dest)
        except Exception as exc:
            return ExportResult(success=False, error=str(exc))

    @staticmethod
    def export_formats() -> List[str]:
        """Return supported export file extensions."""
        return [".png", ".svg", ".pdf"]


# ═══════════════════════════════════════════════════════════════════════
#  FilterableTableManager — data-only logic for filterable tables
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TableState:
    """Tracks the data and view state for a single table."""

    source_df: Optional[pd.DataFrame] = None
    view_df: Optional[pd.DataFrame] = None
    sort_col: Optional[str] = None
    sort_asc: bool = True
    filter_text: str = ""


class FilterableTableManager:
    """Pure-logic table management — no widgets.

    Holds ``TableState`` per table key and provides filtering, sorting
    and column-width estimation.
    """

    def __init__(self) -> None:
        self._tables: Dict[str, TableState] = {}

    # ── Registration ────────────────────────────────────────────

    def register(self, key: str) -> TableState:
        """Register a new table identified by *key* and return its freshly created ``TableState``.

        If *key* already exists its state is silently replaced.
        """
        state = TableState()
        self._tables[key] = state
        return state

    def get_state(self, key: str) -> Optional[TableState]:
        """Return the ``TableState`` for *key*, or ``None``."""
        return self._tables.get(key)

    @property
    def keys(self) -> List[str]:
        """Return all registered table keys."""
        return list(self._tables.keys())

    # ── Data ────────────────────────────────────────────────────

    def set_data(self, key: str, df: Optional[pd.DataFrame]) -> None:
        """Replace the source DataFrame for *key* and reset filter, sort, and view state."""
        state = self._tables.get(key)
        if state is None:
            return
        state.source_df = df
        state.view_df = df
        state.filter_text = ""
        state.sort_col = None
        state.sort_asc = True

    # ── Filtering ───────────────────────────────────────────────

    def apply_filter(self, key: str, text: str) -> Optional[pd.DataFrame]:
        """Apply a case-insensitive substring filter across all columns.

        Updates the internal view DataFrame and returns it, or ``None``
        if the table key is unregistered or has no source data.
        """
        state = self._tables.get(key)
        if state is None or state.source_df is None:
            return None
        state.filter_text = text
        if not text.strip():
            state.view_df = state.source_df
        else:
            needle = text.strip().lower()
            mask = state.source_df.apply(
                lambda row: any(
                    needle in str(v).lower() for v in row
                ),
                axis=1,
            )
            state.view_df = state.source_df[mask]
        return state.view_df

    # ── Sorting ─────────────────────────────────────────────────

    def toggle_sort(self, key: str, column: str) -> Optional[pd.DataFrame]:
        """Toggle ascending/descending sort on *column* for the table at *key*.

        On first click the column sorts ascending; subsequent clicks on
        the same column flip direction.  Returns the sorted view DataFrame
        or ``None`` if the key is unregistered.
        """
        state = self._tables.get(key)
        if state is None or state.view_df is None:
            return None
        if column not in state.view_df.columns:
            return state.view_df
        if state.sort_col == column:
            state.sort_asc = not state.sort_asc
        else:
            state.sort_col = column
            state.sort_asc = True
        try:
            state.view_df = state.view_df.sort_values(
                by=column,
                ascending=state.sort_asc,
                na_position="last",
            )
        except TypeError:
            state.view_df = state.view_df.sort_values(
                by=column,
                ascending=state.sort_asc,
                key=lambda s: s.astype(str),
                na_position="last",
            )
        return state.view_df

    # ── Column width estimation ─────────────────────────────────

    @staticmethod
    def estimate_column_width(
        col_name: str,
        values: Sequence[Any],
        *,
        char_width: int = 8,
        min_width: int = 70,
        max_width: int = 350,
        sample_size: int = 50,
    ) -> int:
        """Estimate pixel width for a column from a sample of values."""
        sample = list(values)[:sample_size]
        lengths = [len(str(v)) for v in sample] + [len(col_name)]
        max_len = max(lengths) if lengths else len(col_name)
        return max(min_width, min(max_width, max_len * char_width + 20))

    # ── Counts ──────────────────────────────────────────────────

    def row_counts(self, key: str) -> Tuple[int, int]:
        """Return *(visible, total)* row counts for *key*."""
        state = self._tables.get(key)
        if state is None:
            return (0, 0)
        total = len(state.source_df) if state.source_df is not None else 0
        visible = len(state.view_df) if state.view_df is not None else 0
        return (visible, total)


# ═══════════════════════════════════════════════════════════════════════
#  StyledOptionMenuHelper — pure-logic for building option menu data
# ═══════════════════════════════════════════════════════════════════════

class StyledOptionMenuHelper:
    """Static helper for preparing option-menu value lists."""

    @staticmethod
    def numeric_columns(df: Optional[pd.DataFrame]) -> List[str]:
        """Return the list of numeric column names from *df*."""
        if df is None or df.empty:
            return []
        return list(df.select_dtypes(include=["number"]).columns)

    @staticmethod
    def sample_names(
        data: Dict[str, Any],
        *,
        sort: bool = True,
    ) -> List[str]:
        """Return sorted sample keys from a results dict."""
        names = list(data.keys())
        if sort:
            names.sort()
        return names

    @staticmethod
    def unique_series_bases(
        df: Optional[pd.DataFrame],
        arquivo_col: str = "Arquivo",
    ) -> List[str]:
        """Extract unique base names from the 'Arquivo' column.

        Base name is everything after the first numeric token, e.g.
        ``"1 NF H2SO4"`` → ``"NF H2SO4"``.
        """
        if df is None or arquivo_col not in df.columns:
            return []
        bases: List[str] = []
        seen: set = set()
        for raw in df[arquivo_col].dropna().unique():
            text = str(raw).replace(".txt", "").strip()
            parts = text.split()
            if not parts:
                continue
            try:
                float(parts[0])
                base = " ".join(parts[1:]) if len(parts) > 1 else text
            except ValueError:
                base = text
            if base and base not in seen:
                seen.add(base)
                bases.append(base)
        return sorted(bases)
