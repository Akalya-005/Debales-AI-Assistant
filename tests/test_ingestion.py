from app.ingestion import canonicalize_url, clean_html, pages_to_documents, CrawledPage


def test_canonicalize_keeps_same_domain():
    url = canonicalize_url("/blog", "https://debales.ai/")
    assert url == "https://debales.ai/blog/"


def test_canonicalize_rejects_external_domain():
    url = canonicalize_url("https://example.com/blog", "https://debales.ai/")
    assert url is None


def test_clean_html_removes_scripts_and_keeps_text():
    html = """
    <html>
      <head><title>Debales Test</title><script>bad()</script></head>
      <body><h1>Hello</h1><p>Debales AI automates logistics work.</p></body>
    </html>
    """
    page = clean_html(html, "https://debales.ai/test")
    assert page.title == "Debales Test"
    assert "Debales AI automates logistics work." in page.text
    assert "bad()" not in page.text


def test_pages_to_documents_adds_metadata():
    docs = pages_to_documents(
        [CrawledPage(url="https://debales.ai/", title="Home", text="Debales AI text")]
    )
    assert docs[0].metadata["source_url"] == "https://debales.ai/"
    assert docs[0].metadata["title"] == "Home"
    assert docs[0].metadata["source_type"] == "debales_website"
