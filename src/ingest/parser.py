"""HTML → clean text parsing for SEC filings."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Strip tags, scripts, styles, and tables; collapse whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def parse_filing(path: Path) -> str:
    """Read an HTML filing from disk and return cleaned text."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return html_to_text(raw)
