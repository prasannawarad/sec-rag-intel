from pathlib import Path

import pytest

from src.ingest.sgml import extract_primary_document, parse_header

SAMPLE_HEADER = """<SEC-DOCUMENT>0000320193-25-000079.txt : 20251031
<SEC-HEADER>0000320193-25-000079.hdr.sgml : 20251031
<ACCEPTANCE-DATETIME>20251031060126
ACCESSION NUMBER:		0000320193-25-000079
CONFORMED SUBMISSION TYPE:	10-K
PUBLIC DOCUMENT COUNT:		91
CONFORMED PERIOD OF REPORT:	20250927
FILED AS OF DATE:		20251031
DATE AS OF CHANGE:		20251031

FILER:

	COMPANY DATA:
		COMPANY CONFORMED NAME:			Apple Inc.
		CENTRAL INDEX KEY:			0000320193
		STANDARD INDUSTRIAL CLASSIFICATION:	ELECTRONIC COMPUTERS [3571]
		ORGANIZATION NAME:           	06 Technology
		EIN:				942404110
		STATE OF INCORPORATION:			CA
		FISCAL YEAR END:			0927
"""


def test_parse_header_extracts_metadata(tmp_path: Path):
    p = tmp_path / "full-submission.txt"
    p.write_text(SAMPLE_HEADER + "\n" * 50)
    h = parse_header(p)
    assert h.accession_number == "0000320193-25-000079"
    assert h.cik == "0000320193"
    assert h.company_name == "Apple Inc."
    assert h.form_type == "10-K"
    assert h.filed_date.isoformat() == "2025-10-31"
    assert h.period_of_report.isoformat() == "2025-09-27"
    assert h.fiscal_year == 2025


def test_parse_header_raises_on_missing_field(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("garbage\n" * 60)
    with pytest.raises(ValueError, match="missing field"):
        parse_header(p)


def test_extract_primary_document_picks_target_type(tmp_path: Path):
    sgml = (
        SAMPLE_HEADER
        + "\n<DOCUMENT>\n<TYPE>10-K\n<FILENAME>main.htm\n<TEXT>"
        + "<html><body><h1>Item 1A. Risk Factors</h1></body></html>"
        + "</TEXT>\n</DOCUMENT>\n"
        + "<DOCUMENT>\n<TYPE>EX-21.1\n<FILENAME>ex.htm\n<TEXT>"
        + "<html>exhibit content</html>"
        + "</TEXT>\n</DOCUMENT>\n"
    )
    p = tmp_path / "full-submission.txt"
    p.write_text(sgml)
    body = extract_primary_document(p, "10-K")
    assert "Item 1A" in body
    assert "exhibit content" not in body
