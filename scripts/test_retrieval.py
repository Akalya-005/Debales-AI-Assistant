
from app.config import Settings
from app.rag import DebalesRetriever
import time

def test_retrieval():
    settings = Settings.from_env()
    print("Initializing retriever...")
    start = time.time()
    retriever = DebalesRetriever(settings)
    print(f"Retriever initialized in {time.time() - start:.2f}s")
    
    query = "What does debales AI automate?"
    print(f"Retrieving for query: {query}")
    start = time.time()
    results = retriever.retrieve(query)
    print(f"Retrieved {len(results)} results in {time.time() - start:.2f}s")
    for res in results:
        print(f"- {res['title']} ({res['score']:.4f})")

if __name__ == "__main__":
    test_retrieval()
