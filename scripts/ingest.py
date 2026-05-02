"""Ingest Debales AI website content into Chroma."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.ingestion import build_vector_store


def main() -> None:
    settings = Settings.from_env()
    print(f"Crawling {settings.debales_base_url}")
    count = build_vector_store(settings, reset=True)
    print(
        f"Ingested {count} chunks into Chroma collection "
        f"{settings.chroma_collection!r} at {settings.chroma_dir}"
    )


if __name__ == "__main__":
    main()
