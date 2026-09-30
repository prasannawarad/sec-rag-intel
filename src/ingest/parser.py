"""Inline-XBRL HTML -> clean text + section-aware splitting for SEC filings.

10-K filings have a highly regular Item structure (Item 1, 1A, 1B, 2, 3, 4, 5,
6, 7, 7A, 8, 9, 9A, 9B, 10-15). Splitting by Item lets citations point to a
specific section ("Risk Factors", "MD&A") instead of the whole filing — that
is what makes a financial RAG system actually useful for analysts.

This module returns a list[Section] where each section carries an item_code
(e.g. "Item 1A") and a human-readable section_label (e.g. "Risk Factors").
Items not in our controlled vocabulary are skipped so retrieval focuses on
the parts analysts actually cite.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from src.ingest.sgml import extract_primary_document

logger = logging.getLogger(__name__)


SECTION_LABELS: dict[str, str] = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 1B": "Unresolved Staff Comments",
    "Item 1C": "Cybersecurity",  # mandatory since FY2023
    "Item 2": "Properties",
    "Item 3": "Legal Proceedings",
    "Item 5": "Market for Registrant's Common Equity",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements",
    "Item 9": "Changes in and Disagreements with Accountants",
    "Item 9A": "Controls and Procedures",
    "Item 9B": "Other Information",
}


# Pseudo item code for a wrapper 10-K's embedded annual report (see _split_by_items).
ANNUAL_REPORT = "Annual Report"
# An Item 7 shorter than this is a cross-reference stub, not a real MD&A.
_WRAPPER_STUB_CHARS = 2_000


@dataclass(frozen=True)
class Section:
    item_code: str
    section_label: str
    text: str


# A heading line: "Item NN[A]" optionally followed by a period and a short
# section label on the same line. Anchored to start/end so we ignore in-prose
# mentions like "see Item 1A" buried inside MD&A. TOC stubs ("Item 1A."), real
# headings ("Item 1A. Risk Factors") and running page headers all match —
# _split_by_items sorts out which is which.
_ITEM_HEADING_LINE_RE = re.compile(
    r"^\s*(?:ITEM|Item)\s+(\d{1,2}[A-Z]?)\.?(?:\s+[A-Za-z][^\n]{0,120})?\s*$",
    re.MULTILINE,
)

_WHITESPACE_RE = re.compile(r"[ \t]+")
# Common unicode spaces that show up in inline-XBRL: NBSP, thin space, en/em quad,
# narrow no-break space, etc. Normalise them all to plain space.
_UNICODE_SPACES = "       　"


# Running page footers, e.g. "Apple Inc. | 2025 Form 10-K | 12" — pure noise for retrieval.
_PAGE_FOOTER_RE = re.compile(r"^.{0,80}\|\s*(?:19|20)\d{2}\s+Form\s+10-[KQ]\s*\|\s*\d+$")


def html_to_text(html: str) -> str:
    """Strip scripts/styles/hidden XBRL facts; keep <table> content (modern
    inline-XBRL filings put real prose inside table cells for layout)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:hidden"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    for ch in _UNICODE_SPACES:
        text = text.replace(ch, " ")
    # Collapse intra-line whitespace but preserve line breaks so headings stay
    # detectable as their own lines.
    lines = [_WHITESPACE_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln and not _PAGE_FOOTER_RE.match(ln)).strip()


def _split_by_items(text: str) -> list[Section]:
    """Slice the document at Item headings that appear on their own line.

    Every Item heading is a boundary, even ones we don't keep (Item 4, 6,
    10–16) — otherwise their text (e.g. the exhibit index) is swallowed by
    the preceding kept section. Consecutive headings for the same item are
    running page headers (MSFT repeats "Item 8" on ~40 pages), so they're
    collapsed into one section. Of the remaining occurrences per item code
    (TOC entry vs. actual section) the LONGEST is kept.
    """
    candidates: list[tuple[int, str]] = []
    for m in _ITEM_HEADING_LINE_RE.finditer(text):
        item_code = f"Item {m.group(1)}"
        if candidates and candidates[-1][1] == item_code:
            continue  # running page header — still inside the same section
        candidates.append((m.start(), item_code))

    sections: list[Section] = []
    for i, (start, item_code) in enumerate(candidates):
        if item_code not in SECTION_LABELS:
            continue  # boundary only (Part III/IV, reserved items)
        end = candidates[i + 1][0] if i + 1 < len(candidates) else len(text)
        body = text[start:end].strip()
        if len(body) < 500:
            continue  # TOC stub; not a real section
        # Drop heading lines (incl. running page headers) so they don't pollute chunks.
        body = _ITEM_HEADING_LINE_RE.sub("", body).lstrip(" .:-\n").strip()
        if not body:
            continue
        sections.append(
            Section(
                item_code=item_code,
                section_label=SECTION_LABELS[item_code],
                text=body,
            )
        )

    by_item: dict[str, Section] = {}
    for s in sections:
        if s.item_code not in by_item or len(s.text) > len(by_item[s.item_code].text):
            by_item[s.item_code] = s

    mdna = by_item.get("Item 7")
    has_mdna_heading = any(code == "Item 7" for _, code in candidates)
    if has_mdna_heading and (mdna is None or len(mdna.text) < _WRAPPER_STUB_CHARS):
        # 10-K "wrapper" (e.g. JPM): Items 7/8 only cross-reference an annual
        # report that is included in the filing *outside* the Item structure —
        # before the first heading and after the last. Keep it, or the filing's
        # MD&A and financial statements are lost.
        outside = [text[: candidates[0][0]].strip(), text[candidates[-1][0] :].strip()]
        body = "\n\n".join(part for part in outside if len(part) >= 500)
        if body:
            by_item[ANNUAL_REPORT] = Section(ANNUAL_REPORT, ANNUAL_REPORT, body)
    return list(by_item.values())


def parse_filing(path: Path, form_type: str = "10-K") -> list[Section]:
    """End-to-end: SGML envelope -> primary doc -> clean text -> sectioned."""
    body = extract_primary_document(path, form_type)
    text = html_to_text(body)
    return _split_by_items(text)
