# Demo Explanation Script

1. "This assistant uses LangGraph to route each question before answering."
2. "For Debales AI questions, it retrieves evidence from a Chroma vector database built from debales.ai."
3. "For general or external questions, it calls SerpAPI."
4. "For mixed questions, it combines Debales evidence with web search evidence."
5. "If evidence is missing, it says 'I don't know' instead of guessing."
6. Show the ingestion command: `python scripts/ingest.py`.
7. Show the CLI startup command: `python -m app.cli`.
8. Ask: "What does Debales AI automate?"
9. Ask: "What is AI in logistics?"
10. Ask: "Compare Debales AI with general logistics automation."
11. Ask: "What is Debales AI's unreleased pricing plan?"
12. Point out the route, citations, and grounded refusal behavior.
