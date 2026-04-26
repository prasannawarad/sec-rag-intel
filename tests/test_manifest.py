from datetime import date
from pathlib import Path

from src.ingest.manifest import Manifest
from src.ingest.sgml import FilingHeader


def _header(accession: str = "0000320193-25-000079") -> FilingHeader:
    return FilingHeader(
        accession_number=accession,
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-K",
        filed_date=date(2025, 10, 31),
        period_of_report=date(2025, 9, 27),
    )


def test_manifest_lifecycle(tmp_path: Path):
    m = Manifest(path=tmp_path / "manifest.parquet")
    assert m.df().is_empty()
    assert not m.has("0000320193-25-000079")

    m.upsert_filing(_header(), ticker="AAPL", raw_path=tmp_path / "raw.txt")
    assert m.has("0000320193-25-000079")
    assert len(m.pending("parse")) == 1
    assert len(m.pending("embed")) == 1

    m.mark_parsed("0000320193-25-000079", num_chunks=42)
    assert len(m.pending("parse")) == 0
    assert len(m.pending("embed")) == 1

    m.mark_embedded("0000320193-25-000079", embed_model="bge-small")
    assert len(m.pending("embed")) == 0

    # Reload from disk: state survives
    m2 = Manifest(path=tmp_path / "manifest.parquet")
    row = m2.df().row(0, named=True)
    assert row["num_chunks"] == 42
    assert row["embed_model"] == "bge-small"
    assert row["fiscal_year"] == 2025


def test_manifest_upsert_replaces_existing_row(tmp_path: Path):
    m = Manifest(path=tmp_path / "manifest.parquet")
    m.upsert_filing(_header(), ticker="AAPL", raw_path=tmp_path / "v1.txt")
    m.upsert_filing(_header(), ticker="AAPL", raw_path=tmp_path / "v2.txt")
    assert len(m.df()) == 1
    assert m.df().row(0, named=True)["raw_path"].endswith("v2.txt")
