"""Offline bibliographic references for IonFlow Pipeline.

Parses ``tutoriais/08_referencias_bibliograficas.txt`` once and exposes
structured ``Reference`` objects for the GUI Help window and the
Streamlit dashboard References page.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

_REFS_FILE = (
    Path(__file__).resolve().parent.parent
    / "tutoriais"
    / "08_referencias_bibliograficas.txt"
)

_DOI_RE = re.compile(r"DOI:\s*([\S]+)", re.IGNORECASE)
_TAG_RE = re.compile(r"^\s*\[([A-Z0-9\-]+)\]")
_PART_LINE_RE = re.compile(r"^PARTE\s+[IVXLC]+\s+", re.IGNORECASE)


class Reference(NamedTuple):
    """Single bibliographic reference entry."""

    tag: str  # e.g. "EIS-1"
    text: str  # full block text as it appears in the source file
    doi: str  # DOI string, e.g. "10.1002/0471716243", or ""
    doi_url: str  # clickable URL "https://doi.org/<doi>", or ""
    part: str  # section header, e.g. "PARTE I — EIS Fundamentos"


def load_references() -> list[Reference]:
    """Parse the references file and return structured Reference objects."""
    if not _REFS_FILE.exists():
        return []

    raw = _REFS_FILE.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Pass 1 — collect (part_name, line_index_start) boundaries
    part_boundaries: list[tuple[str, int]] = [("CABEÇALHO", 0)]
    i = 0
    while i < len(lines):
        if re.match(r"={40,}", lines[i]):
            # Look ahead for a PARTE line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _PART_LINE_RE.match(lines[j].strip()):
                part_title = lines[j].strip()
                # Remove trailing "===..." and "Tutoriais xx, yy" sub-lines
                part_title = re.sub(r"\s+Tutoriais.*", "", part_title)
                part_boundaries.append((part_title, i))
        i += 1
    part_boundaries.append(("__END__", len(lines)))

    # Pass 2 — parse each part section for reference blocks
    refs: list[Reference] = []
    seen_tags: set[str] = set()

    for idx, (part_name, start_line) in enumerate(part_boundaries[:-1]):
        end_line = part_boundaries[idx + 1][1]
        section_lines = lines[start_line:end_line]

        # Split section into blocks that start with [TAG]
        blocks: list[list[str]] = []
        current_block: list[str] = []
        for line in section_lines:
            if _TAG_RE.match(line) and current_block:
                blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)

        for block_lines in blocks:
            block_text = "\n".join(block_lines).strip()
            m = _TAG_RE.match(block_lines[0])
            if not m:
                continue
            tag = m.group(1)
            if tag in seen_tags:
                continue
            seen_tags.add(tag)

            doi_match = _DOI_RE.search(block_text)
            doi = doi_match.group(1).rstrip(".,)") if doi_match else ""
            doi_url = f"https://doi.org/{doi}" if doi else ""

            refs.append(
                Reference(
                    tag=tag,
                    text=block_text,
                    doi=doi,
                    doi_url=doi_url,
                    part=part_name,
                )
            )

    return refs


def filter_references(refs: list[Reference], query: str) -> list[Reference]:
    """Return refs whose tag, text or part header contains *query* (case-insensitive)."""
    if not query.strip():
        return refs
    q = query.strip().lower()
    return [
        r
        for r in refs
        if q in r.tag.lower() or q in r.text.lower() or q in r.part.lower()
    ]


def get_raw_text() -> str:
    """Return the full references file as a plain string."""
    if _REFS_FILE.exists():
        return _REFS_FILE.read_text(encoding="utf-8")
    return "(Arquivo de referências não encontrado)"
