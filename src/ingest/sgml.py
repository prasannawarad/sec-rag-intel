"""Parse SEC EDGAR full-submission SGML files.

The downloader saves each filing as a single full-submission.txt with an SGML
header followed by one or more <DOCUMENT> blocks. The header has the metadata
we need for the ingestion manifest (accession number, fiscal period, filing
date, company name). The first DOCUMENT block of TYPE 10-K / 10-Q contains
the inline-XBRL HTML body.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilingHeader:
    accession_number: str
    cik: str
    company_name: str
    form_type: str
    filed_date: date
    period_of_report: date

    @property
    def fiscal_year(self) -> int:
        return self.period_of_report.year


_HEADER_PATTERNS = {
    "accession_number": re.compile(r"ACCESSION NUMBER:\s+(\S+)"),
    "cik": re.compile(r"CENTRAL INDEX KEY:\s+(\S+)"),
    "company_name": re.compile(r"COMPANY CONFORMED NAME:\s+(.+?)\n"),
    "form_type": re.compile(r"CONFORMED SUBMISSION TYPE:\s+(\S+)"),
    "filed_date": re.compile(r"FILED AS OF DATE:\s+(\d{8})"),
    "period_of_report": re.compile(r"CONFORMED PERIOD OF REPORT:\s+(\d{8})"),
}


def parse_header(path: Path) -> FilingHeader:
    """Extract structured metadata from the SGML header (first ~50 lines)."""
    with path.open(encoding="utf-8", errors="ignore") as f:
        head = "".join(next(f) for _ in range(60))

    fields: dict[str, str] = {}
    for name, pattern in _HEADER_PATTERNS.items():
        m = pattern.search(head)
        if not m:
            raise ValueError(f"SGML header in {path} missing field: {name}")
        fields[name] = m.group(1).strip()

    return FilingHeader(
        accession_number=fields["accession_number"],
        cik=fields["cik"],
        company_name=fields["company_name"],
        form_type=fields["form_type"],
        filed_date=datetime.strptime(fields["filed_date"], "%Y%m%d").date(),
        period_of_report=datetime.strptime(fields["period_of_report"], "%Y%m%d").date(),
    )


_DOCUMENT_RE = re.compile(
    r"<DOCUMENT>\s*<TYPE>(?P<type>\S+).*?<TEXT>(?P<body>.*?)</TEXT>\s*</DOCUMENT>",
    re.DOTALL,
)


def extract_primary_document(path: Path, target_type: str) -> str:
    """Return the inline-XBRL HTML body of the first DOCUMENT matching target_type.

    target_type examples: "10-K", "10-Q". Exhibits (EX-*) are ignored.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    for match in _DOCUMENT_RE.finditer(text):
        if match.group("type").strip() == target_type:
            return match.group("body").strip()
    raise ValueError(f"No <DOCUMENT> with TYPE={target_type} in {path}")
