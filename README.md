# 🚀 Debales AI: Intelligent RAG-Powered Logistics Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, production-grade AI chatbot designed to answer questions about **Debales AI** with high precision. It uses an intelligent routing system to switch between internal knowledge (RAG) and external web search (SerpAPI), ensuring every answer is grounded in evidence and citation-backed.

---

## 📌 Problem Statement
General-purpose LLMs often hallucinate specific company details or lack the latest information from niche websites. For a logistics-tech company like Debales AI, providing inaccurate information about integrations, workflows (3PL, Freight, Carrier), or the "Messometer" can lead to confusion.

## 💡 Solution Overview
This project implements a **Retreival-Augmented Generation (RAG)** pipeline using **LangGraph** to build a deterministic workflow.
- **Debales Queries:** Answered via a localized ChromaDB vector store.
- **General Queries:** Answered via live Google Search (SerpAPI).
- **Mixed Queries:** Synthesized from both internal documents and web results.

## ⚙️ Key Features
- **Intelligent Query Routing:** Automatically detects if a query is about Debales, the general industry, or both.
- **Zero Hallucination Policy:** Strict grounding ensures the LLM says "I don't know" if evidence isn't found.
- **Self-Healing Ingestion:** Crawls `debales.ai` automatically, cleaning HTML and managing chunks.
- **Hybrid Embeddings:** Supports both OpenAI and local deterministic embeddings.
- **Interactive CLI:** Professional terminal interface with color-coded routing logs and citations.

## 🧠 Architecture
The system follows a modular "Chain-of-Thought" graph architecture:

1. **Input Node:** Normalizes user query.
2. **Intent Classifier:** Routes query based on keywords and semantic context.
3. **Retrieval Engine:** Parallel execution of RAG (Chroma) and Web Search (SerpAPI).
4. **Response Generator:** Synthesizes the final answer with mandatory source citations.

## 🛠️ Tech Stack
- **LLM Orchestration:** LangChain, LangGraph
- **Vector Database:** ChromaDB
- **LLM Providers:** OpenAI (GPT-4o-mini)
- **Search Engine:** SerpAPI (Google Search)
- **Data Processing:** BeautifulSoup4, RecursiveCharacterTextSplitter
- **Testing:** Pytest

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Akalya-005/Debales-AI-Assistant.git
cd Debales-AI-Assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file from the example:
```bash
cp .env.example .env
```
Fill in your API keys in the `.env` file:
- `OPENAI_API_KEY`: For LLM and Embeddings.
- `SERPAPI_API_KEY`: For live web search.

### 3. Run Ingestion
Build your local knowledge base by crawling the Debales AI website:
```bash
python scripts/ingest.py
```

### 4. Launch the Assistant
```bash
python -m app.cli
```

---

## 📸 Sample Outputs

**Example 1: Internal RAG Query**
> **User:** What does Debales AI automate?
> **Route:** `debales`
> **Answer:** Debales AI automates logistics workflows including freight booking, 3PL integrations, and carrier communications. It specifically focuses on reducing manual data entry by 80% through its AI agents.
> **Sources:** [https://debales.ai/features]

**Example 2: External Industry Query**
> **User:** What is the current trend in AI for logistics?
> **Route:** `external`
> **Answer:** Current trends in AI for logistics include predictive demand forecasting, autonomous last-mile delivery, and the use of computer vision for warehouse sorting...
> **Sources:** [https://www.forbes.com/logistics-trends-2024]

---

## 🛡️ No Hallucination Strategy
To ensure recruiter-ready reliability, this project implements:
1. **System Prompt Constraint:** LLM is strictly instructed NOT to use its own knowledge for company-specific claims.
2. **Relevance Filtering:** Chunks with a low similarity score (< 0.20) are discarded before generation.
3. **Citation Requirement:** Every claim must be followed by a source URL from the retrieved evidence.

## 📈 Future Improvements
- [ ] Add a Web-based UI (Next.js/Streamlit).
- [ ] Implement Multi-turn conversation memory.
- [ ] Add support for local LLMs (Ollama/Llama 3).

## 👩‍💻 Author
**Akalya**
- GitHub: https://github.com/Akalya-005
- LinkedIn: https://www.linkedin.com/in/akalya-k-548937294?utm_source=share_via&utm_content=profile&utm_medium=member_android
