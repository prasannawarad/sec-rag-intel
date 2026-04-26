from src.ingest.parser import html_to_text


def test_html_to_text_strips_tags_and_scripts():
    html = """
    <html><head><style>body { color: red }</style></head>
    <body><script>alert('x')</script>
    <h1>Risk Factors</h1>
    <p>Our supply chain is concentrated.</p>
    <table><tr><td>ignored</td></tr></table>
    </body></html>
    """
    text = html_to_text(html)
    assert "Risk Factors" in text
    assert "supply chain" in text
    assert "alert" not in text
    assert "ignored" not in text
