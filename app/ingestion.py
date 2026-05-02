"""Crawl Debales AI pages and persist them into Chroma."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.config import Settings
from app.embeddings import get_embeddings


ASSET_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".mp4",
    ".mov",
    ".webm",
    ".xml",
}

SKIP_SCHEMES = {"mailto", "tel", "sms", "javascript"}


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str
    text: str


def canonicalize_url(raw_url: str, base_url: str) -> str | None:
    """Normalize and filter URLs to the Debales domain."""

    if not raw_url:
        return None
    joined = urljoin(base_url, raw_url)
    joined, _fragment = urldefrag(joined)
    parsed = urlparse(joined)
    base = urlparse(base_url)

    if parsed.scheme in SKIP_SCHEMES:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() != base.netloc.lower():
        return None
    if Path(parsed.path).suffix.lower() in ASSET_EXTENSIONS:
        return None
    if parsed.query:
        return None

    path = parsed.path or "/"
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    url = urlunparse(normalized)
    if url.endswith("/") or Path(path).suffix:
        return url
    return f"{url}/"


def fetch_url(session: requests.Session, url: str, timeout: int = 15, retries: int = 2) -> str:
    """Fetch a page with small retry/backoff handling."""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def clean_html(html: str, source_url: str) -> CrawledPage:
    """Extract readable text while preserving useful page structure."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "iframe",
            "header",
            "footer",
            "nav",
        ]
    ):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else source_url

    lines: list[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "td", "th"]):
        text = tag.get_text(" ", strip=True)
        if text:
            lines.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        compact = " ".join(line.split())
        if len(compact) < 2 or compact in seen:
            continue
        seen.add(compact)
        deduped.append(compact)

    return CrawledPage(url=source_url, title=title, text="\n".join(deduped))


def extract_links(html: str, page_url: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        url = canonicalize_url(anchor["href"], page_url)
        if url and canonicalize_url(url, base_url):
            links.append(url)
    return links


def crawl_site(settings: Settings) -> list[CrawledPage]:
    """Breadth-first crawl of Debales pages."""

    start_url = canonicalize_url(settings.debales_base_url, settings.debales_base_url)
    if start_url is None:
        raise ValueError(f"Invalid DEBALES_BASE_URL: {settings.debales_base_url}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "DebalesAIAssistant/0.1 "
                "(RAG ingestion; contact support@debales.ai)"
            )
        }
    )

    queue = [start_url]
    seen: set[str] = set()
    pages: list[CrawledPage] = []

    while queue and len(pages) < settings.crawl_max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            html = fetch_url(session, url)
        except RuntimeError as exc:
            print(f"[warn] {exc}")
            continue
        if not html:
            continue

        page = clean_html(html, url)
        if len(page.text) >= 100:
            pages.append(page)

        for link in extract_links(html, url, settings.debales_base_url):
            if link not in seen and link not in queue:
                queue.append(link)

    return pages


def pages_to_documents(pages: Iterable[CrawledPage]) -> list[Document]:
    crawl_time = datetime.now(timezone.utc).isoformat()
    documents = []
    for page in pages:
        documents.append(
            Document(
                page_content=page.text,
                metadata={
                    "source_url": page.url,
                    "title": page.title,
                    "crawl_time": crawl_time,
                    "source_type": "debales_website",
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    return splitter.split_documents(documents)


def build_vector_store(settings: Settings, reset: bool = True) -> int:
    """Crawl, chunk, embed, and persist Debales AI content."""

    pages = crawl_site(settings)
    if not pages:
        raise RuntimeError("No Debales pages were crawled. Check DEBALES_BASE_URL/network access.")

    documents = pages_to_documents(pages)
    chunks = split_documents(documents)
    if not chunks:
        raise RuntimeError("No chunks were produced from crawled pages.")

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(settings)

    vector_store = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    if reset:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    vector_store.add_documents(chunks)
    return len(chunks)
