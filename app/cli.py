"""Command-line chatbot interface."""

from __future__ import annotations

from app.config import Settings
from app.graph import DebalesAssistant


def _print_response(result: dict) -> None:
    answer = result.get('answer', "I don't have enough information to answer this.")
    print(f"\nAnswer: {answer}\n")
    print(f"Route: {result.get('intent', 'unknown')}\n")

    sources = result.get("sources") or []
    if sources:
        print("Sources:")
        for source in sources:
            title = source.get("title") or "Source"
            url = source.get("url") or ""
            source_type = source.get("source_type") or "source"
            print(f"* [{source_type}] {title} - {url}")
    print()


def main() -> None:
    settings = Settings.from_env()
    assistant = DebalesAssistant(settings=settings)

    print("Debales AI Assistant")
    print("Ask about Debales AI or general logistics topics. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if not query:
            continue

        result = assistant.ask(query)
        _print_response(result)


if __name__ == "__main__":
    main()
