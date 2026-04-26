from src.ingest.parser import _split_by_items, html_to_text


def test_html_to_text_strips_scripts_and_styles_keeps_table_content():
    """Modern inline-XBRL filings put real prose in <table> cells, so we keep them."""
    html = """
    <html><head><style>body { color: red }</style></head>
    <body><script>alert('x')</script>
    <h1>Item 1A. Risk Factors</h1>
    <p>Our supply chain is concentrated.</p>
    <table><tr><td>Important risk content in a table cell.</td></tr></table>
    </body></html>
    """
    text = html_to_text(html)
    assert "Risk Factors" in text
    assert "supply chain" in text
    assert "Important risk content" in text  # table content preserved
    assert "alert" not in text
    assert "color: red" not in text


def test_html_to_text_normalises_unicode_spaces():
    html = "<p>Item\xa01A. Risk Factors</p>"
    text = html_to_text(html)
    assert "Item 1A." in text


def test_split_by_items_picks_real_section_over_toc_stub():
    """Both 'Item 1A.' (TOC) and 'Item 1A. Risk Factors' (real heading) appear
    on their own lines — the longer body must win."""
    text = (
        "Item 1.\nItem 1A.\nItem 1B.\nItem 2.\n"  # TOC block
        + "Item 1. Business\n"
        + ("Apple makes phones. " * 60)
        + "\n"
        + "Item 1A. Risk Factors\n"
        + ("Supply chain concentration risk. " * 200)
        + "\n"
        + "Item 1B. Unresolved Staff Comments\n"
        + ("None. " * 100)
        + "\n"
    )
    sections = _split_by_items(text)
    by_code = {s.item_code: s for s in sections}
    assert "Item 1A" in by_code
    assert "Risk Factors" in by_code["Item 1A"].section_label
    assert "Supply chain concentration risk" in by_code["Item 1A"].text
    # TOC stub did NOT win
    assert len(by_code["Item 1A"].text) > 1000


def test_split_by_items_ignores_in_prose_mentions():
    """A 'see Item 1A' reference inside MD&A must not be detected as a heading."""
    text = (
        "Item 7. Management's Discussion and Analysis\n"
        + "We discuss this further; see Item 1A above for more detail. "
        + "Revenue grew. " * 100
    )
    sections = _split_by_items(text)
    by_code = {s.item_code: s for s in sections}
    assert "Item 7" in by_code
    # Item 1A appears only inside Item 7 prose, not as a heading line
    assert "Item 1A" not in by_code


def test_split_by_items_skips_unknown_item_codes():
    text = (
        "Item 99. Something Made Up\n"
        + ("filler " * 200)
        + "\n"
        + "Item 1A. Risk Factors\n"
        + ("real risk content. " * 200)
        + "\n"
    )
    sections = _split_by_items(text)
    codes = [s.item_code for s in sections]
    assert "Item 99" not in codes
    assert "Item 1A" in codes
